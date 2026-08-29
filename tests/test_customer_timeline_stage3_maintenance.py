from __future__ import annotations

import json
import hashlib
import sqlite3
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import mango_mvp.customer_timeline.stage3_maintenance as stage3_module

from mango_mvp.customer_timeline import (
    BotContextChunk,
    CustomerIdentity,
    CustomerOpportunity,
    CustomerTimelineSQLiteStore,
    DerivedSignal,
    IdentityLink,
    Stage3MaintenanceConfig,
    TimelineDirection,
    TimelineEvent,
    TimelineEventType,
    run_stage3_maintenance,
)
from mango_mvp.customer_timeline.ids import stable_digest


NOW = datetime(2026, 7, 3, 12, 0, tzinfo=timezone.utc)


def _identity() -> CustomerIdentity:
    return CustomerIdentity(
        tenant_id="foton",
        identity_status="strong",
        display_name="Тестовый клиент",
        primary_phone="+79161234567",
        primary_email="client@example.com",
        first_seen_at=NOW,
        last_seen_at=NOW,
        touch_count=1,
        created_at=NOW,
        updated_at=NOW,
    )


def _email_event(customer: CustomerIdentity | None, *, source_id: str, preview: str) -> TimelineEvent:
    return TimelineEvent(
        tenant_id="foton",
        customer_id=customer.customer_id if customer else None,
        event_type=TimelineEventType.EMAIL_MESSAGE,
        event_at=NOW,
        source_system="mail_archive_stage2",
        source_id=source_id,
        direction=TimelineDirection.INBOUND,
        subject="Заявка с сайта",
        text_preview=preview,
        summary="Клиент уточнил расписание группы и попросил ответить.",
        importance=2,
        match_status="strong_unique" if customer else "unmatched",
        confidence=0.9 if customer else None,
        created_at=NOW,
        record={"message_sha256": source_id},
    )


def _mail_chunk(event: TimelineEvent, *, text: str) -> BotContextChunk:
    return BotContextChunk(
        tenant_id=event.tenant_id,
        customer_id=event.customer_id or "",
        event_id=event.event_id,
        source_ref=event.source_ref,
        source_system=event.source_system,
        chunk_type="email_message",
        text=text,
        summary=event.summary or "",
        event_at=event.event_at,
        freshness_score=0.7,
        relevance_tags=("email",),
        allowed_for_bot=False,
        requires_manager_review=True,
        created_at=event.created_at,
    )


def test_stage3_soft_deletes_only_attributed_duplicates_and_keeps_fts_clean(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    with CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path) as store:
        customer = _identity()
        store.upsert_customer(customer)
        first = _email_event(customer, source_id="a" * 64, preview="Клиент спрашивает расписание.")
        duplicate = replace(
            _email_event(customer, source_id="b" * 64, preview="Клиент спрашивает расписание повторно."),
            created_at=NOW + timedelta(seconds=1),
        )
        same_key_different_preview = replace(
            _email_event(customer, source_id="f" * 64, preview="Клиент просит подобрать другую группу."),
            created_at=NOW + timedelta(seconds=1),
        )
        none_first = _email_event(None, source_id="c" * 64, preview="Безымянная web-форма 1.")
        none_second = replace(
            _email_event(None, source_id="d" * 64, preview="Безымянная web-форма 2."),
            created_at=NOW + timedelta(seconds=1),
        )
        store.upsert_event(first)
        store.upsert_event(duplicate)
        store.upsert_event(same_key_different_preview)
        store.upsert_event(none_first)
        store.upsert_event(none_second)
        store.upsert_bot_context_chunk(_mail_chunk(first, text="уникальныйдубль первый"))
        store.upsert_bot_context_chunk(_mail_chunk(duplicate, text="уникальныйдубль второй"))
        store.upsert_bot_context_chunk(_mail_chunk(same_key_different_preview, text="не дубль по preview"))
        content_key = store._con.execute(  # noqa: SLF001 - test fixture prepares historical duplicate rows.
            "SELECT content_key FROM timeline_events WHERE event_id = ?",
            (first.event_id,),
        ).fetchone()[0]
        store._con.execute(  # noqa: SLF001
            "UPDATE timeline_events SET content_key = ? WHERE event_id IN (?, ?, ?)",
            (content_key, same_key_different_preview.event_id, none_first.event_id, none_second.event_id),
        )
        store._con.execute(  # noqa: SLF001
            "UPDATE timeline_events SET content_key = ?, text_preview = ? WHERE event_id = ?",
            (content_key, first.text_preview, duplicate.event_id),
        )
        store._con.commit()  # noqa: SLF001

    report = run_stage3_maintenance(
        Stage3MaintenanceConfig(
            timeline_db_path=db_path,
            allowed_root=tmp_path,
            out_dir=tmp_path / "out",
            apply=True,
        )
    )

    assert report["duplicate_plan"]["groups"] == 1
    assert report["duplicate_plan"]["duplicate_events"] == 1
    assert report["duplicate_plan"]["none_customer_groups_report_only"]["groups"] == 1
    assert report["duplicate_plan"]["mixed_preview_groups_report_only"] == 1
    assert report["soft_delete"]["superseded_events"] == 1
    assert report["soft_delete"]["superseded_chunks"] == 1
    assert report["final_checks"]["fts_superseded_counts"] == {
        "timeline_event_fts_superseded": 0,
        "timeline_event_fts_keys_superseded": 0,
        "bot_context_chunk_fts_superseded": 0,
    }
    assert report["validation_ok"] is True

    with sqlite3.connect(db_path) as con:
        hidden = con.execute(
            "SELECT superseded_by FROM timeline_events WHERE event_id = ?",
            (duplicate.event_id,),
        ).fetchone()[0]
        none_hidden = con.execute(
            "SELECT count(*) FROM timeline_events WHERE event_id IN (?, ?) AND superseded_by IS NOT NULL",
            (none_first.event_id, none_second.event_id),
        ).fetchone()[0]
        different_preview_hidden = con.execute(
            "SELECT superseded_by FROM timeline_events WHERE event_id = ?",
            (same_key_different_preview.event_id,),
        ).fetchone()[0]
        mail_allowed = con.execute(
            "SELECT count(*) FROM bot_context_chunks WHERE source_system = 'mail_archive_stage2' AND allowed_for_bot != 0"
        ).fetchone()[0]
    assert hidden == first.event_id
    assert none_hidden == 0
    assert different_preview_hidden is None
    assert mail_allowed == 0


def test_stage3_chunk_label_backfill_is_conservative_for_raw_mail(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    with CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path) as store:
        customer = _identity()
        store.upsert_customer(customer)
        event = _email_event(customer, source_id="e" * 64, preview="Письмо про расписание.")
        store.upsert_event(event)
        store.upsert_bot_context_chunk(_mail_chunk(event, text="Текст письма"))

    report = run_stage3_maintenance(
        Stage3MaintenanceConfig(
            timeline_db_path=db_path,
            allowed_root=tmp_path,
            out_dir=tmp_path / "out",
            apply=True,
        )
    )

    assert report["chunk_label_backfill"]["counts"]["chunks_updated"] == 1
    with sqlite3.connect(db_path) as con:
        row = con.execute("SELECT record_json FROM bot_context_chunks").fetchone()
    payload = __import__("json").loads(row[0])
    assert payload["allowed_for_bot"] is False
    assert payload["requires_manager_review"] is True
    assert payload["metadata"]["client_safe"] is False
    assert payload["metadata"]["client_safe_reason"] == "stage2_mail_manager_review_pending"
    assert payload["metadata"]["client_safe_policy_version"] == "cs_v1"
    assert payload["metadata"]["memory_status"] == "manager_review_required"


def test_stage3_hardens_legacy_mail_chunk_columns_and_json(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    with CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path) as store:
        customer = _identity()
        store.upsert_customer(customer)
        event = _email_event(customer, source_id="9" * 64, preview="Старое письмо.")
        store.upsert_event(event)
        store.upsert_bot_context_chunk(_mail_chunk(event, text="Полный текст старого письма"))

    with sqlite3.connect(db_path) as con:
        row = con.execute("SELECT chunk_id, record_json FROM bot_context_chunks").fetchone()
        payload = json.loads(row[1])
        payload["allowed_for_bot"] = True
        payload["requires_manager_review"] = False
        con.execute(
            """
            UPDATE bot_context_chunks
            SET allowed_for_bot = 1, requires_manager_review = 0, record_json = ?
            WHERE chunk_id = ?
            """,
            (json.dumps(payload, ensure_ascii=False), row[0]),
        )
        con.commit()

    config = Stage3MaintenanceConfig(
        timeline_db_path=db_path,
        allowed_root=tmp_path,
        out_dir=tmp_path / "out",
        apply=True,
    )
    report = run_stage3_maintenance(config)
    repeated = run_stage3_maintenance(config)

    assert report["mail_stage2_visibility_hardening"]["updated_chunks"] == 1
    assert report["fts_rebuild"] == {"performed": True, "reasons": ["mail_stage2_visibility", "chunk_labels"]}
    assert repeated["mail_stage2_visibility_hardening"]["updated_chunks"] == 0
    assert repeated["chunk_label_backfill"]["counts"].get("chunks_updated", 0) == 0
    assert repeated["fts_rebuild"] == {"performed": False, "reasons": []}
    assert report["after"]["mail_stage2_chunks_allowed"] == 0
    assert report["after"]["mail_stage2_chunks_without_review"] == 0
    with sqlite3.connect(db_path) as con:
        allowed, review, record_json = con.execute(
            "SELECT allowed_for_bot, requires_manager_review, record_json FROM bot_context_chunks"
        ).fetchone()
    payload = json.loads(record_json)
    assert (allowed, review) == (0, 1)
    assert payload["allowed_for_bot"] is False
    assert payload["requires_manager_review"] is True


def test_stage3_repairs_only_exact_mail_identity_sentinel_dates_and_is_idempotent(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    sentinel = datetime(1970, 1, 1, tzinfo=timezone.utc)
    with CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path) as store:
        first = _identity()
        second = replace(
            _identity(),
            customer_id="customer:second",
            primary_phone="+79160000002",
            primary_email="second@example.com",
        )
        store.upsert_customer(first)
        store.upsert_customer(second)

        def add_case(
            suffix: str,
            *,
            owner: CustomerIdentity = first,
            match_class: str = "strong_unique",
            first_seen_at: datetime = sentinel,
            last_seen_at: datetime = sentinel,
            event_owners: tuple[CustomerIdentity, ...] = (first,),
        ) -> str:
            source_ref = f"mail-exact:{suffix}"
            link = IdentityLink(
                tenant_id="foton",
                customer_id=owner.customer_id,
                link_type="tallanto_student_id",
                link_value=f"student-{suffix}",
                source_system="mail_archive_stage2",
                source_ref=source_ref,
                match_class=match_class,
                confidence=0.9,
                first_seen_at=first_seen_at,
                last_seen_at=last_seen_at,
            )
            store.upsert_identity_link(link)
            for index, event_owner in enumerate(event_owners):
                event = replace(
                    _email_event(
                        event_owner,
                        source_id=f"{suffix}-{index}".ljust(64, "0")[:64],
                        preview=f"Письмо {suffix} {index}",
                    ),
                    source_ref=source_ref,
                    event_at=NOW + timedelta(minutes=index),
                    created_at=NOW + timedelta(minutes=index),
                )
                store.upsert_event(event)
            return str(link.link_id)

        strong_id = add_case("strong")
        ambiguous_id = add_case("ambiguous", match_class="ambiguous")
        one_field_id = add_case("one-field", last_seen_at=NOW + timedelta(days=1))
        missing_id = add_case("owner-mismatch", event_owners=(second,))
        multi_id = add_case("multi", event_owners=(first, first))

    config = Stage3MaintenanceConfig(
        timeline_db_path=db_path,
        allowed_root=tmp_path,
        out_dir=tmp_path / "out-repair",
        apply=True,
        signal_as_of=NOW + timedelta(days=10),
    )
    first_report = run_stage3_maintenance(config)
    second_report = run_stage3_maintenance(config)

    summary = first_report["mail_identity_date_repair_plan"]
    assert summary["links_scanned"] == 5
    assert summary["links_actionable"] == 5
    assert summary["links_exact_date"] == 3
    assert summary["links_date_cleared_unknown"] == 2
    assert summary["missing_exact_evidence"] == 1
    assert summary["ambiguous_exact_evidence"] == 1
    assert summary["match_class_counts_preserved"] == {"ambiguous": 1, "strong_unique": 4}
    assert first_report["mail_identity_date_repair"]["links_repaired"] == 5
    assert second_report["mail_identity_date_repair_plan"]["links_scanned"] == 0
    assert second_report["mail_identity_date_repair"]["links_repaired"] == 0

    with sqlite3.connect(db_path) as con:
        con.row_factory = sqlite3.Row
        rows = {
            row["link_id"]: row
            for row in con.execute(
                "SELECT link_id,match_class,first_seen_at,last_seen_at,record_hash,record_json FROM identity_links"
            )
        }
        audit_count = con.execute(
            "SELECT count(*) FROM audit_log WHERE action='identity_link_seen_at_repaired'"
        ).fetchone()[0]
    for link_id in (strong_id, ambiguous_id, one_field_id):
        payload = json.loads(rows[link_id]["record_json"])
        assert rows[link_id]["record_hash"] == stable_digest(payload)
        assert payload["evidence"]["mail_stage2_date_repair"]["reason"] == "legacy_epoch_sentinel"
    assert rows[strong_id]["first_seen_at"] == NOW.isoformat()
    assert rows[strong_id]["last_seen_at"] == NOW.isoformat()
    assert rows[ambiguous_id]["match_class"] == "ambiguous"
    assert rows[one_field_id]["last_seen_at"] == (NOW + timedelta(days=1)).isoformat()
    for link_id, reason in ((missing_id, "missing_exact_evidence"), (multi_id, "ambiguous_exact_evidence")):
        payload = json.loads(rows[link_id]["record_json"])
        assert rows[link_id]["first_seen_at"] is None
        assert rows[link_id]["last_seen_at"] is None
        assert payload["evidence"]["mail_stage2_date_repair"] == {
            "reason": "legacy_epoch_sentinel",
            "resolution": "source_date_unknown",
            "unknown_reason": reason,
        }
    assert audit_count == 5


@pytest.mark.parametrize(
    "damage",
    ("non_sentinel", "wrong_source"),
)
def test_direct_mail_identity_date_repair_rejects_out_of_contract_link(
    tmp_path: Path,
    damage: str,
) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    sentinel = datetime(1970, 1, 1, tzinfo=timezone.utc)
    with CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path) as store:
        customer = _identity()
        store.upsert_customer(customer)
        source_ref = "mail-proof:direct"
        source_system = "tallanto_export" if damage == "wrong_source" else "mail_archive_stage2"
        link = IdentityLink(
            tenant_id="foton",
            customer_id=customer.customer_id,
            link_type="tallanto_student_id",
            link_value=f"direct-{damage}",
            source_system=source_system,
            source_ref=source_ref,
            match_class="strong_unique",
            first_seen_at=NOW if damage == "non_sentinel" else sentinel,
            last_seen_at=NOW if damage == "non_sentinel" else sentinel,
        )
        store.upsert_identity_link(link)
        exact = replace(
            _email_event(customer, source_id="e" * 64, preview="Exact evidence"),
            source_system=source_system,
            source_ref=source_ref,
        )
        store.upsert_event(exact)
        before = store._con.execute(  # noqa: SLF001
            "SELECT first_seen_at,last_seen_at,record_hash,record_json FROM identity_links WHERE link_id=?",
            (link.link_id,),
        ).fetchone()
        audit_before = store._con.execute(  # noqa: SLF001
            "SELECT count(*) FROM audit_log WHERE action='identity_link_seen_at_repaired'"
        ).fetchone()[0]

        with pytest.raises(ValueError):
            store.repair_identity_link_seen_at(
                "foton",
                link_id=str(link.link_id),
                as_of=NOW + timedelta(days=1),
            )

        after = store._con.execute(  # noqa: SLF001
            "SELECT first_seen_at,last_seen_at,record_hash,record_json FROM identity_links WHERE link_id=?",
            (link.link_id,),
        ).fetchone()
        audit_after = store._con.execute(  # noqa: SLF001
            "SELECT count(*) FROM audit_log WHERE action='identity_link_seen_at_repaired'"
        ).fetchone()[0]
        assert tuple(after) == tuple(before)
        assert audit_after == audit_before


def test_stage3_mail_identity_date_repair_rolls_back_as_one_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    sentinel = datetime(1970, 1, 1, tzinfo=timezone.utc)
    with CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path) as store:
        customer = _identity()
        store.upsert_customer(customer)
        for index in range(2):
            source_ref = f"mail-rollback:{index}"
            store.upsert_identity_link(
                IdentityLink(
                    tenant_id="foton",
                    customer_id=customer.customer_id,
                    link_type="tallanto_student_id",
                    link_value=f"rollback-student-{index}",
                    source_system="mail_archive_stage2",
                    source_ref=source_ref,
                    match_class="strong_unique",
                    first_seen_at=sentinel,
                    last_seen_at=sentinel,
                )
            )
            store.upsert_event(
                replace(
                    _email_event(customer, source_id=str(index).zfill(64), preview=f"Rollback {index}"),
                    source_ref=source_ref,
                )
            )

    original = CustomerTimelineSQLiteStore.repair_identity_link_seen_at
    calls = 0

    def fail_second(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("repair fault")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(CustomerTimelineSQLiteStore, "repair_identity_link_seen_at", fail_second)
    with pytest.raises(RuntimeError, match="repair fault"):
        run_stage3_maintenance(
            Stage3MaintenanceConfig(
                timeline_db_path=db_path,
                allowed_root=tmp_path,
                out_dir=tmp_path / "out-rollback",
                apply=True,
            )
        )

    with sqlite3.connect(db_path) as con:
        assert con.execute(
            "SELECT count(*) FROM identity_links WHERE first_seen_at=? AND last_seen_at=?",
            (sentinel.isoformat(), sentinel.isoformat()),
        ).fetchone()[0] == 2
        assert con.execute(
            "SELECT count(*) FROM audit_log WHERE action='identity_link_seen_at_repaired'"
        ).fetchone()[0] == 0


def test_stage3_rejects_case_insensitive_prod_path_before_writing(tmp_path: Path) -> None:
    db_path = tmp_path / "CUSTOMER_TIMELINE_PROD_TEST" / "customer_timeline.sqlite"

    with pytest.raises(ValueError, match="snapshot-only"):
        run_stage3_maintenance(
            Stage3MaintenanceConfig(
                timeline_db_path=db_path,
                allowed_root=tmp_path,
                out_dir=tmp_path / "out",
                apply=True,
            )
        )

    assert not db_path.exists()


def test_stage3_calls_preflight_fails_before_any_timeline_write(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    calls_db = tmp_path / "unsupported_calls.sqlite"
    with CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path) as store:
        customer = _identity()
        store.upsert_customer(customer)
        store.upsert_event(_email_event(customer, source_id="1" * 64, preview="До preflight"))
    with sqlite3.connect(calls_db) as con:
        con.execute("CREATE TABLE unsupported_calls (id INTEGER PRIMARY KEY)")
    before_sha = hashlib.sha256(db_path.read_bytes()).hexdigest()

    with pytest.raises(ValueError, match="exactly one supported source table"):
        run_stage3_maintenance(
            Stage3MaintenanceConfig(
                timeline_db_path=db_path,
                allowed_root=tmp_path,
                out_dir=tmp_path / "out-preflight",
                canonical_calls_db_path=calls_db,
                apply=True,
            )
        )

    assert hashlib.sha256(db_path.read_bytes()).hexdigest() == before_sha
    assert not (tmp_path / "out-preflight").exists()


def test_stage3_removes_nullish_identity_shell_and_preserves_quarantined_raw_event(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    with CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path) as store:
        customer = _identity()
        store.upsert_customer(customer)
        link = IdentityLink(
            tenant_id="foton",
            customer_id=customer.customer_id,
            link_type="amo_contact_id",
            link_value="contact-nullish",
            source_system="amocrm_snapshot",
            source_ref="contact:nullish",
            confidence=1.0,
        )
        store.upsert_identity_link(link)
        opportunity = CustomerOpportunity(
            tenant_id="foton",
            customer_id=customer.customer_id,
            opportunity_type="amo_deal",
            source_system="amocrm_snapshot",
            source_id="lead-nullish",
            opened_at=NOW,
        )
        store.upsert_opportunity(opportunity)
        raw_event = replace(
            _email_event(
                customer,
                source_id="nullish-raw".ljust(64, "0"),
                preview="Сырое событие",
            ),
            opportunity_id=opportunity.opportunity_id,
            subject="сохранённоесыроесобытие",
            summary="сохранённоесыроесобытие",
        )
        store.upsert_event(raw_event)
        context = replace(
            _mail_chunk(raw_event, text="производныйконтекст удалить"),
            opportunity_id=opportunity.opportunity_id,
        )
        store.upsert_bot_context_chunk(context)
        signal = DerivedSignal(
            tenant_id="foton",
            customer_id=customer.customer_id,
            opportunity_id=opportunity.opportunity_id,
            event_id=raw_event.event_id,
            source_event_ids=(raw_event.event_id,),
            signal_type="follow_up",
            severity="low",
            evidence_text="Производный сигнал",
            created_at=NOW,
        )
        store.upsert_signal(signal)
        family_payload = {
            "tenant_id": "foton",
            "family_id": "family:nullish",
            "customer_id": "None",
            "membership_status": "singleton",
            "confidence": "high",
            "reason": "test",
            "created_at": NOW.isoformat(),
            "updated_at": NOW.isoformat(),
        }
        store._con.execute(  # noqa: SLF001 - historical corruption fixture.
            "INSERT INTO family_members_v1 VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                "foton",
                "family:nullish",
                "None",
                "singleton",
                "high",
                "test",
                NOW.isoformat(),
                NOW.isoformat(),
                stable_digest(family_payload),
                json.dumps(family_payload, ensure_ascii=False),
            ),
        )
        for table in (
            "customer_identities",
            "identity_links",
            "customer_opportunities",
            "timeline_events",
            "derived_signals",
            "bot_context_chunks",
        ):
            rows = store._con.execute(  # noqa: SLF001
                f"SELECT rowid,record_json FROM {table} WHERE customer_id=?", (customer.customer_id,)
            ).fetchall()
            for rowid, record_json in rows:
                payload = json.loads(record_json)
                payload["customer_id"] = "None"
                store._con.execute(  # noqa: SLF001
                    f"UPDATE {table} SET customer_id='None',record_json=?,record_hash=? WHERE rowid=?",
                    (
                        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                        stable_digest(payload),
                        rowid,
                    ),
                )
        store._con.commit()  # noqa: SLF001

    dry_config = Stage3MaintenanceConfig(
        timeline_db_path=db_path,
        allowed_root=tmp_path,
        out_dir=tmp_path / "out-dry",
        apply=False,
        signal_as_of=NOW + timedelta(days=1),
    )
    dry = run_stage3_maintenance(dry_config)
    assert dry["nullish_customer_id_repair_plan"] == {
        "bad_literal_values": 1,
        "literal_counts": {
            "bot_context_chunks.customer_id": 1,
            "customer_identities.customer_id": 1,
            "customer_opportunities.customer_id": 1,
            "derived_signals.customer_id": 1,
            "family_members_v1.customer_id": 1,
            "identity_links.customer_id": 1,
            "timeline_events.customer_id": 1,
        },
        "active_events": 1,
        "identity_links": 1,
        "opportunities": 1,
        "derived_signals": 1,
        "bot_context_chunks": 1,
        "family_members": 1,
        "customer_identities": 1,
        "stop_counts": {},
    }
    with sqlite3.connect(db_path) as con:
        assert con.execute("SELECT COUNT(*) FROM customer_identities WHERE customer_id='None'").fetchone()[0] == 1

    apply_config = replace(dry_config, out_dir=tmp_path / "out-apply", apply=True)
    applied = run_stage3_maintenance(apply_config)
    repeated = run_stage3_maintenance(apply_config)
    assert applied["nullish_customer_id_repair"] == {
        "writes": 7,
        "events_quarantined": 1,
        "links_detached": 1,
        "opportunities_deleted": 1,
        "signals_deleted": 1,
        "chunks_deleted": 1,
        "family_members_deleted": 1,
        "customer_identities_deleted": 1,
    }
    assert repeated["nullish_customer_id_repair"]["writes"] == 0
    assert applied["validation_ok"] is True
    assert repeated["validation_ok"] is True

    with CustomerTimelineSQLiteStore.open_read_only(db_path, allowed_root=tmp_path) as store:
        con = store._con  # noqa: SLF001
        event_row = con.execute(
            "SELECT customer_id,opportunity_id,match_status,confidence,record_json "
            "FROM timeline_events WHERE event_id=?",
            (raw_event.event_id,),
        ).fetchone()
        assert tuple(event_row[:4]) == (None, None, "ambiguous", 0.0)
        assert json.loads(event_row[4])["metadata"]["resolution_reason"] == "textual_null_customer_id"
        assert con.execute("SELECT COUNT(*) FROM identity_links WHERE customer_id IS NULL").fetchone()[0] == 1
        assert con.execute("SELECT COUNT(*) FROM customer_opportunities").fetchone()[0] == 0
        assert con.execute("SELECT COUNT(*) FROM derived_signals").fetchone()[0] == 0
        assert con.execute("SELECT COUNT(*) FROM bot_context_chunks").fetchone()[0] == 0
        assert con.execute("SELECT COUNT(*) FROM family_members_v1").fetchone()[0] == 0
        assert con.execute("SELECT COUNT(*) FROM customer_identities").fetchone()[0] == 0
        assert store.search_timeline("foton", "сохранённоесыроесобытие", mode="fts")["items"] == []
        assert store.search_timeline("foton", "сохранённоесыроесобытие", mode="fallback")["items"] == []


def test_stage3_stops_before_writing_on_mixed_nullish_source_pair(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    with CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path) as store:
        first = _identity()
        second = replace(
            _identity(),
            customer_id="customer:second",
            primary_phone="+79160000002",
            primary_email="second@example.com",
        )
        store.upsert_customer(first)
        store.upsert_customer(second)
        bad = _email_event(first, source_id="mixed-source".ljust(64, "0"), preview="bad")
        good = replace(bad, customer_id=second.customer_id, event_type="system_note", event_id=None)
        store.upsert_event(bad)
        store.upsert_event(good)
        payload = bad.to_json_dict()
        payload["customer_id"] = "None"
        store._con.execute(  # noqa: SLF001
            "UPDATE timeline_events SET customer_id='None',record_json=?,record_hash=? WHERE event_id=?",
            (
                json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                stable_digest(payload),
                bad.event_id,
            ),
        )
        store._con.commit()  # noqa: SLF001

    with pytest.raises(RuntimeError, match="mixed_active_source_pairs"):
        run_stage3_maintenance(
            Stage3MaintenanceConfig(
                timeline_db_path=db_path,
                allowed_root=tmp_path,
                out_dir=tmp_path / "out-mixed",
                apply=True,
            )
        )
    with sqlite3.connect(db_path) as con:
        assert con.execute("SELECT COUNT(*) FROM timeline_events WHERE customer_id='None'").fetchone()[0] == 1
        assert con.execute("SELECT COUNT(*) FROM audit_log WHERE actor='stage3_nullish_customer_id_repair'").fetchone()[0] == 0
