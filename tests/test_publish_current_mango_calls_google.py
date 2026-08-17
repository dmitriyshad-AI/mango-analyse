from __future__ import annotations

import json
import os
import shutil
import sqlite3
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from mango_mvp.productization import owner_only_io
from tests.conftest import dual_strict_source, ready_capture_proof

from mango_mvp.productization.mango_calls_service_contract import (
    STAGE10_SCHEMA,
    approved_runtime_fingerprint,
    sha256_file,
)
from mango_mvp.productization.ready_publication import (
    commit_ready_generation,
    inspect_ready_publication,
)
from mango_mvp.models import CallRecord
from mango_mvp.services.dialogue_contract import manager_output_sha256
from scripts import publish_current_mango_calls_google as publisher
from tests import mango_provider_fixture as fx
from tests.test_ai_office_export import valid_v3_analysis


OWNER = "owner@example.test"
SERVICE = "calls@example.test"
ROP = "rop@example.test"


def _interrupt_ready_publication(ready_db: Path) -> None:
    staged = ready_db.parent / "synthetic-next.sqlite"
    shutil.copy2(ready_db, staged)
    staged.chmod(0o600)
    with sqlite3.connect(staged) as con:
        con.execute("CREATE TABLE synthetic_publication_marker(value INTEGER)")
    manifest_path = ready_db.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        ready_db=str(ready_db),
        sha256=sha256_file(staged),
        size_bytes=staged.stat().st_size,
        ready_mtime_ns=staged.stat().st_mtime_ns,
    )

    def crash(stage: str) -> None:
        if stage == "db_replaced":
            raise RuntimeError("synthetic ready publication crash")

    with pytest.raises(RuntimeError, match="synthetic ready publication crash"):
        commit_ready_generation(ready_db, staged, manifest, checkpoint=crash)


class NoAccessGateway:
    def __init__(self) -> None:
        self.accesses: list[str] = []

    def __getattr__(self, name: str) -> object:
        self.accesses.append(name)
        raise AssertionError(f"legacy publisher touched gateway attribute {name}")


def manager_row(key: str, *, summary: str = "Обсудили программу обучения.") -> dict[str, object]:
    return {
        "call_key": key,
        "Дата и время": "11.08.2026 12:00:00",
        "Менеджер": "Менеджер",
        "Направление": "Входящий",
        "Клиент": "Клиент",
        "Телефон": "+70000000000",
        "Длительность": 60.0,
        "Тип разговора": "Продажа",
        "Краткое содержание": summary,
        "Результат": "Назначен следующий контакт",
        "Интерес клиента": "Курс",
        "Главное возражение": "Стоимость",
        "Следующий шаг": "Перезвонить",
        "Срок": "12.08.2026",
        publisher.TRANSCRIPT_LINK_HEADER: "",
        "Нужна проверка": "Нет",
        "Причина проверки": "",
    }


def test_credentials_policy_uses_path_of_exact_opened_inode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cloud = tmp_path / "Yandex.Disk" / "credentials.json"
    cloud.parent.mkdir()
    cloud.write_text(json.dumps({"client_email": "cloud@example.test"}), encoding="utf-8")
    cloud.chmod(0o600)
    local = tmp_path / "local.json"
    local.write_text(json.dumps({"client_email": "local@example.test"}), encoding="utf-8")
    local.chmod(0o600)
    original = publisher.read_stable_regular_bytes_with_path

    def swap_after_read(path: Path, **kwargs: object) -> tuple[bytes, Path]:
        raw, opened_path = original(path, **kwargs)
        path.unlink()
        path.symlink_to(local)
        return raw, opened_path

    monkeypatch.setattr(publisher, "read_stable_regular_bytes_with_path", swap_after_read)
    with pytest.raises(RuntimeError, match="outside repository and cloud"):
        publisher.validate_credentials(cloud)


def test_credentials_reject_hardlink_with_cloud_alias(tmp_path: Path) -> None:
    local = tmp_path / "local" / "credentials.json"
    local.parent.mkdir()
    local.write_text(json.dumps({"client_email": "linked@example.test"}), encoding="utf-8")
    local.chmod(0o600)
    cloud = tmp_path / "Yandex.Disk" / "credentials.json"
    cloud.parent.mkdir()
    os.link(local, cloud)

    with pytest.raises(RuntimeError, match="owner-only 0600"):
        publisher.validate_credentials(local)


def test_credentials_reject_parent_move_during_descriptor_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local_root = tmp_path / "local"
    local_root.mkdir()
    credentials = local_root / "credentials.json"
    credentials.write_text(json.dumps({"client_email": "moved@example.test"}), encoding="utf-8")
    credentials.chmod(0o600)
    cloud_root = tmp_path / "Yandex.Disk"
    cloud_root.mkdir()
    moved_root = cloud_root / "local"
    original_read = owner_only_io.os.read
    swapped = False

    def move_parent_then_read(descriptor: int, size: int) -> bytes:
        nonlocal swapped
        if not swapped:
            local_root.rename(moved_root)
            local_root.symlink_to(moved_root, target_is_directory=True)
            swapped = True
        return original_read(descriptor, size)

    monkeypatch.setattr(owner_only_io.os, "read", move_parent_then_read)
    with pytest.raises(RuntimeError, match="owner-only 0600"):
        publisher.validate_credentials(credentials)
    assert swapped is True


def test_credentials_reject_macos_cloudstorage_path(tmp_path: Path) -> None:
    credentials = (
        tmp_path / "Library" / "CloudStorage" / "GoogleDrive-test" / "credentials.json"
    )
    credentials.parent.mkdir(parents=True)
    credentials.write_text(json.dumps({"client_email": "cloud@example.test"}), encoding="utf-8")
    credentials.chmod(0o600)

    with pytest.raises(RuntimeError, match="outside repository and cloud"):
        publisher.validate_credentials(credentials)


def test_legacy_publish_function_refuses_before_google_access() -> None:
    gateway = NoAccessGateway()

    with pytest.raises(
        RuntimeError,
        match=r"plan-only; use publish_live_mango_calls_google\.py",
    ):
        publisher.publish_current(
            gateway,
            day=date(2026, 8, 11),
            desired_rows=[manager_row("call-1")],
            owner_email=OWNER,
            allowed_emails=(SERVICE, ROP),
            pilot_started_day=date(2026, 8, 1),
            retention_approved=False,
            prior_day=None,
        )

    assert gateway.accesses == []


def test_plan_only_module_exposes_no_google_write_gateway() -> None:
    gateway = publisher.GoogleGateway(object(), "spreadsheet-id")
    assert not hasattr(gateway, "batch_sheet_requests")
    assert not hasattr(gateway, "write_values")
    assert not hasattr(gateway, "clear_values")
    assert not hasattr(publisher, "plan_upsert")
    assert not hasattr(publisher, "verify_readback")


# The planner reads the very same stored payloads as the live publisher, so it
# applies the very same fail-closed role guard.  These fixtures are the proven
# case; ``roles_proven=False`` is the real production majority and has its own
# test below.
READY_TURNS = (
    ("client", "right", "Нас интересует математика для 9 класса."),
    ("operator", "left", "Хорошо, я отправлю программу."),
)


def _ready_fixture(
    tmp_path: Path, *, analyzed: bool, roles_proven: bool = True
) -> tuple[Path, Path]:
    db = tmp_path / "ready.sqlite"
    with sqlite3.connect(db) as con:
        con.execute(
            """
            CREATE TABLE call_records (
                id INTEGER, source_call_id TEXT, started_at TEXT, manager_name TEXT,
                phone TEXT, direction TEXT, duration_sec REAL,
                transcription_status TEXT, resolve_status TEXT, analysis_status TEXT,
                analysis_json TEXT, transcript_variants_json TEXT,
                transcript_text TEXT, source_file TEXT, source_recording_id TEXT
            )
            """
        )
        variants = {
            "mode": "stereo",
            "primary_provider": "mlx",
            "secondary_provider": "gigaam",
            "manager": {"variant_a": "Здравствуйте", "variant_b": "Здравствуйте"},
            "client": {"variant_a": "Нужен курс", "variant_b": "Нужен курс"},
            "dialogue_lines": fx.dialogue_lines(READY_TURNS),
        }
        if roles_proven:
            variants["role_mapping"] = dict(fx.PROVEN_ROLE_MAPPING)
            variants[fx.PROVIDER_EVIDENCE_FIELD] = fx.evidence(
                READY_TURNS, source_call_id="call-1"
            )
        fixture_call = CallRecord(
            id=1,
            source_call_id="call-1",
            source_recording_id=fx.RECORDING_ID,
            started_at=datetime(2026, 8, 11, 9, tzinfo=timezone.utc),
            manager_name="Менеджер",
            phone="+70000000000",
            direction="inbound",
            duration_sec=45,
            transcript_variants_json=json.dumps(variants, ensure_ascii=False),
            transcript_text="СЕКРЕТНЫЙ ПОЛНЫЙ ДИАЛОГ",
            source_file="/Users/private/call.mp3",
            source_filename="call.mp3",
        )
        analysis = valid_v3_analysis(fixture_call) if analyzed else {}
        if analyzed:
            analysis["quality_flags"].update(
                call_type="sales_call",
                needs_review=True,
                review_reasons=["synthetic_review"],
            )
            analysis["needs_review"] = True
            analysis["review_reasons"] = ["synthetic_review"]
            analysis["analysis_meta"]["manager_output_sha256"] = (
                manager_output_sha256(analysis)
            )
        con.execute(
            "INSERT INTO call_records VALUES (1,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                "call-1",
                "2026-08-11T09:00:00+00:00",
                "Менеджер",
                "+70000000000",
                "inbound",
                45,
                "done",
                "done",
                "done" if analyzed else "pending",
                json.dumps(analysis, ensure_ascii=False),
                json.dumps(variants, ensure_ascii=False),
                "СЕКРЕТНЫЙ ПОЛНЫЙ ДИАЛОГ",
                "/Users/private/call.mp3",
                fx.RECORDING_ID,
            ),
        )
    verdict = {
        "schema_version": STAGE10_SCHEMA,
        "quarantine_items": [],
        "day": "2026-08-11",
        "generated_at": "2026-08-11T10:00:00+00:00",
        "mango_enumeration_complete": True,
        "mango_unique": 1,
        "ready_unique": 1,
        "quarantine_unique": 0,
        "pending_unique": 0,
        "unexplained_missing": 0,
        "state_overlap_count": 0,
        "pending_awaiting_recording": 0,
        "pending_over_sla": 0,
        "quarantine_without_reason": 0,
        "ready_without_dual_asr_or_explicit_exception": 0,
        "ready_without_resolve": 0,
        "ready_without_analyze": 0 if analyzed else 1,
        "duplicate_call_keys": 0,
        "oldest_pending_age_minutes": 0,
        "state_not_in_mango_enumeration": 0,
        "independent_zero_enumerations": 0,
        "consistency_ok": True,
        "closure_ok": analyzed,
    }
    enumeration_source = {
        "mode": "strict_service",
        "since": "2026-08-10T21:00:00+00:00",
        "until": "2026-08-11T21:00:00+00:00",
        "cursor": "not_applicable_stats_request_result",
        "pages": None,
        "pagination": "not_applicable_stats_request_result",
        "requests": 1,
        "covered_intervals": [
            {
                "since": "2026-08-10T21:00:00+00:00",
                "until": "2026-08-11T21:00:00+00:00",
                "result_complete": True,
            }
        ],
    }
    enumeration_source = dual_strict_source(
        enumeration_source,
        call_keys=["call-1"],
        calls_by_day={"2026-08-11": ["call-1"]},
    )
    verdict["mango_enumeration_source"] = enumeration_source
    capture_proof, capture_proof_sha256 = ready_capture_proof(
        enumeration_source,
        zero_by_day={"2026-08-11": 0},
    )
    manifest = {
        "schema_version": "mango_calls_ready_v3",
        "created_at_utc": "2026-08-11T10:00:01+00:00",
        "published_at": "2026-08-11T10:00:02+00:00",
        "status": "ready",
        "consistency_ok": True,
        "closure_ok": analyzed,
        "moscow_dates": ["2026-08-11"],
        "daily_verdicts": {"2026-08-11": verdict},
        "producer_git_sha": "a" * 40,
        "host_id": "m1-host",
        "run_id": "run-1",
        "mango_window": {
            "since": "2026-08-10T21:00:00+00:00",
            "until": "2026-08-11T21:00:00+00:00",
        },
        "mango_enumeration_complete": True,
        "mango_enumeration_source": enumeration_source,
        "capture_proof": capture_proof,
        "capture_proof_sha256": capture_proof_sha256,
        "capture_proof_run_id": enumeration_source["dual_enumeration"][
            "proof_run_id"
        ],
        "manifest_snapshot": {"end_offset": 1, "sha256": "b" * 64},
        "provenance_mode": "strict_service",
        "quick_check": "ok",
        "integrity_check": "ok",
        "runtime_fingerprint": approved_runtime_fingerprint(),
        "sha256": sha256_file(db),
        "size_bytes": db.stat().st_size,
    }
    manifest_path = db.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return db, manifest_path


def _rewrite_analysis(
    db: Path,
    manifest_path: Path,
    mutate: object,
) -> None:
    with sqlite3.connect(db) as con:
        payload = json.loads(
            con.execute("SELECT analysis_json FROM call_records").fetchone()[0]
        )
        assert callable(mutate)
        mutate(payload)
        con.execute(
            "UPDATE call_records SET analysis_json=?",
            (json.dumps(payload, ensure_ascii=False),),
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(sha256=sha256_file(db), size_bytes=db.stat().st_size)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def test_projection_never_contains_full_dialogue_path_or_diagnostic_json(tmp_path: Path) -> None:
    db, manifest = _ready_fixture(tmp_path, analyzed=False)
    rows = publisher.load_manager_rows(
        ready_db=db,
        ready_manifest=manifest,
        day=date(2026, 8, 11),
        owner_email=OWNER,
        allowed_emails=(SERVICE, ROP),
    )
    serialized = json.dumps(rows, ensure_ascii=False)
    assert "СЕКРЕТНЫЙ ПОЛНЫЙ ДИАЛОГ" not in serialized
    assert "/Users/" not in serialized and "analysis_json" not in serialized
    assert rows[0]["Краткое содержание"] == publisher.NEUTRAL_SUMMARY
    assert set(rows[0]) == set(publisher.MANAGED_HEADERS)


def test_capture_quarantine_is_visible_as_safe_manager_review_row(
    tmp_path: Path,
) -> None:
    db, manifest_path = _ready_fixture(tmp_path, analyzed=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    verdict = manifest["daily_verdicts"]["2026-08-11"]
    verdict.update(
        mango_unique=2,
        quarantine_unique=1,
        quarantine_items=[
            {
                "call_key": "capture-quarantine-1",
                "started_at": "2026-08-11T09:30:00+00:00",
                "code": "recording_retry_expired",
                "reason": "Аудиозапись не появилась в Mango в течение 72 часов.",
                "action": (
                    "Проверить запись в Mango и повторить загрузку вручную, "
                    "если файл появился."
                ),
            }
        ],
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    rows = publisher.load_manager_rows(
        ready_db=db,
        ready_manifest=manifest_path,
        day=date(2026, 8, 11),
        owner_email=OWNER,
        allowed_emails=(SERVICE, ROP),
    )

    assert len(rows) == 1
    plan = publisher.build_safe_plan(
        day=date(2026, 8, 11),
        rows=rows,
        ready_manifest=manifest,
        now=datetime(2026, 8, 11, 10, tzinfo=timezone.utc),
    )
    assert len(publisher.validate_safe_plan_payload(
        plan,
        expected_day=date(2026, 8, 11),
        now=datetime(2026, 8, 11, 10, 1, tzinfo=timezone.utc),
    )) == 1


def test_private_link_requires_exact_acl_evidence(tmp_path: Path) -> None:
    db, manifest = _ready_fixture(tmp_path, analyzed=True)
    evidence = {
        "call-1": {
            "url": "https://drive.google.com/file/d/private/view",
            "acl_readback_ok": True,
            "allowed_emails": [OWNER, SERVICE, ROP],
        }
    }
    rows = publisher.load_manager_rows(
        ready_db=db,
        ready_manifest=manifest,
        day=date(2026, 8, 11),
        owner_email=OWNER,
        allowed_emails=(SERVICE, ROP),
        link_evidence=evidence,
    )
    assert rows[0][publisher.TRANSCRIPT_LINK_HEADER].startswith("https://")
    evidence["call-1"]["allowed_emails"] = [OWNER, SERVICE, "outsider@example.test"]
    with pytest.raises(RuntimeError, match="ACL"):
        publisher.load_manager_rows(
            ready_db=db,
            ready_manifest=manifest,
            day=date(2026, 8, 11),
            owner_email=OWNER,
            allowed_emails=(SERVICE, ROP),
            link_evidence=evidence,
        )


def test_projection_uses_real_analyze_schema_without_list_repr(tmp_path: Path) -> None:
    db, manifest = _ready_fixture(tmp_path, analyzed=True)
    row = publisher.load_manager_rows(
        ready_db=db,
        ready_manifest=manifest,
        day=date(2026, 8, 11),
        owner_email=OWNER,
        allowed_emails=(SERVICE, ROP),
    )[0]

    assert row["Тип разговора"] == "Продажа / подбор обучения"
    assert row["Интерес клиента"] == ""
    assert row["Главное возражение"] == ""
    assert row["Следующий шаг"] == ""
    assert row["Срок"] == ""
    assert row["Результат"] == "Итог не зафиксирован"
    assert row["Нужна проверка"] == "Да"
    assert "текущей схемой Analyze" not in str(row["Причина проверки"])


def test_the_planner_never_revives_a_role_dependent_field_of_an_unproven_call(
    tmp_path: Path,
) -> None:
    """Planning is still reading, and reading an old payload is not trusting it.

    This script only prepares a plan, but a human copies that plan into Google.
    So it applies the same fail-closed guard as the live publisher: with the
    sides unproven there is no next step, no deadline and no objection.
    """
    db, manifest = _ready_fixture(tmp_path, analyzed=True, roles_proven=False)
    row = publisher.load_manager_rows(
        ready_db=db,
        ready_manifest=manifest,
        day=date(2026, 8, 11),
        owner_email=OWNER,
        allowed_emails=(SERVICE, ROP),
    )[0]

    assert row["Следующий шаг"] == ""
    assert row["Срок"] == ""
    assert row["Главное возражение"] == ""
    assert row["Интерес клиента"] == ""
    assert row["Нужна проверка"] == "Да"
    # The neutral metadata of the call is kept: the row does not disappear.
    assert row["Телефон"] == "'+70000000000"
    assert row["Менеджер"] == "Менеджер"
    serialized = json.dumps(row, ensure_ascii=False)
    assert "Перезвонить после обсуждения" not in serialized
    assert "Стоимость" not in serialized


def test_missing_second_asr_is_explicitly_marked_for_review(tmp_path: Path) -> None:
    db, manifest_path = _ready_fixture(tmp_path, analyzed=True)
    with sqlite3.connect(db) as con:
        raw = con.execute(
            "SELECT transcript_variants_json FROM call_records"
        ).fetchone()[0]
        variants = json.loads(raw)
        variants.pop("secondary_provider")
        for role in ("manager", "client"):
            variants[role].pop("variant_b")
        con.execute(
            "UPDATE call_records SET transcript_variants_json=?",
            (json.dumps(variants, ensure_ascii=False),),
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update({"sha256": sha256_file(db), "size_bytes": db.stat().st_size})
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    row = publisher.load_manager_rows(
        ready_db=db,
        ready_manifest=manifest_path,
        day=date(2026, 8, 11),
        owner_email=OWNER,
        allowed_emails=(SERVICE, ROP),
    )[0]

    assert row["Нужна проверка"] == "Да"
    assert "Вторая расшифровка GigaAM не готова" in str(row["Причина проверки"])
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    plan = publisher.build_safe_plan(
        day=date(2026, 8, 11),
        rows=[row],
        ready_manifest=manifest_payload,
        now=datetime(2026, 8, 11, 10, tzinfo=timezone.utc),
    )
    assert plan["stage10_counts"]["processing_ready_unique"] == 0
    assert plan["row_completion_ok"] is False
    assert plan["closure_ok"] is False


def test_manual_resolve_with_reviewed_analysis_is_processing_ready(tmp_path: Path) -> None:
    db, manifest_path = _ready_fixture(tmp_path, analyzed=True)
    with sqlite3.connect(db) as con:
        con.execute("UPDATE call_records SET resolve_status='manual'")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(sha256=sha256_file(db), size_bytes=db.stat().st_size)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    row = publisher.load_manager_rows(
        ready_db=db,
        ready_manifest=manifest_path,
        day=date(2026, 8, 11),
        owner_email=OWNER,
        allowed_emails=(SERVICE, ROP),
    )[0]

    assert "Разделение ролей не завершено" not in str(row["Причина проверки"])
    assert publisher.processing_ready_row(row) is True


def test_mutated_non_conversation_cannot_override_current_result(tmp_path: Path) -> None:
    db, manifest = _ready_fixture(tmp_path, analyzed=True)

    def mutate(payload: dict[str, object]) -> None:
        payload["quality_flags"] = {"call_type": "non_conversation"}
        payload["needs_review"] = False
        structured = payload["structured_fields"]
        assert isinstance(structured, dict)
        structured["next_step"] = {"action": "", "due": ""}

    _rewrite_analysis(db, manifest, mutate)
    row = publisher.load_manager_rows(
        ready_db=db,
        ready_manifest=manifest,
        day=date(2026, 8, 11),
        owner_email=OWNER,
        allowed_emails=(SERVICE, ROP),
    )[0]

    assert row["Результат"] == "Итог не зафиксирован"
    assert "нужен повторный анализ" in str(row["Причина проверки"])


def test_v2_without_supported_outcome_gets_neutral_result_and_review(
    tmp_path: Path,
) -> None:
    db, manifest = _ready_fixture(tmp_path, analyzed=True)

    def mutate(payload: dict[str, object]) -> None:
        payload["quality_flags"] = {"call_type": "sales_call"}
        payload["needs_review"] = False
        structured = payload["structured_fields"]
        assert isinstance(structured, dict)
        structured["next_step"] = {"action": "", "due": ""}

    _rewrite_analysis(db, manifest, mutate)
    row = publisher.load_manager_rows(
        ready_db=db,
        ready_manifest=manifest,
        day=date(2026, 8, 11),
        owner_email=OWNER,
        allowed_emails=(SERVICE, ROP),
    )[0]

    assert row["Результат"] == "Итог не зафиксирован"
    assert row["Нужна проверка"] == "Да"
    assert "Итог разговора требует ручной проверки" in str(row["Причина проверки"])


def test_legacy_explicit_result_cannot_bypass_current_contract(tmp_path: Path) -> None:
    db, manifest = _ready_fixture(tmp_path, analyzed=True)
    _rewrite_analysis(
        db,
        manifest,
        lambda payload: payload.update(result="Договор отправлен"),
    )

    row = publisher.load_manager_rows(
        ready_db=db,
        ready_manifest=manifest,
        day=date(2026, 8, 11),
        owner_email=OWNER,
        allowed_emails=(SERVICE, ROP),
    )[0]

    assert row["Результат"] == "Итог не зафиксирован"
    assert "Договор отправлен" not in json.dumps(row, ensure_ascii=False)
    assert "нужен повторный анализ" in str(row["Причина проверки"])


@pytest.mark.parametrize("length", [2_001, 32_000])
def test_analyze_summary_is_preserved_without_truncation(
    tmp_path: Path, length: int
) -> None:
    expected = "я" * length
    summary, issue = publisher.manager_summary(expected)

    assert summary == expected
    assert issue == ""


def test_summary_over_analyze_limit_is_replaced_per_row(tmp_path: Path) -> None:
    summary, issue = publisher.manager_summary("я" * 32_001)

    assert summary == publisher.OVERSIZED_SUMMARY
    assert "предел Analyze" in issue


def test_safe_plan_expires_after_sixty_minutes(tmp_path: Path) -> None:
    db, manifest_path = _ready_fixture(tmp_path, analyzed=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = publisher.load_manager_rows(
        ready_db=db,
        ready_manifest=manifest_path,
        ready_manifest_payload=manifest,
        day=date(2026, 8, 11),
        owner_email=OWNER,
        allowed_emails=(SERVICE, ROP),
    )
    generated = datetime(2026, 8, 11, 10, tzinfo=timezone.utc)
    plan = publisher.build_safe_plan(
        day=date(2026, 8, 11),
        rows=rows,
        ready_manifest=manifest,
        now=generated,
    )

    assert publisher.validate_safe_plan_payload(
        plan,
        expected_day=date(2026, 8, 11),
        now=generated + timedelta(minutes=60),
    )
    with pytest.raises(RuntimeError, match="expired"):
        publisher.validate_safe_plan_payload(
            plan,
            expected_day=date(2026, 8, 11),
            now=generated + timedelta(minutes=60, seconds=1),
        )


def test_safe_plan_validator_rejects_rows_removed_after_build(
    tmp_path: Path,
) -> None:
    db, manifest_path = _ready_fixture(tmp_path, analyzed=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = publisher.load_manager_rows(
        ready_db=db,
        ready_manifest=manifest_path,
        ready_manifest_payload=manifest,
        day=date(2026, 8, 11),
        owner_email=OWNER,
        allowed_emails=(SERVICE, ROP),
    )
    generated = datetime(2026, 8, 11, 10, tzinfo=timezone.utc)
    plan = dict(
        publisher.build_safe_plan(
            day=date(2026, 8, 11),
            rows=rows,
            ready_manifest=manifest,
            now=generated,
        )
    )
    plan["rows"] = []

    with pytest.raises(
        RuntimeError, match="row completion state|rows do not match Stage10"
    ):
        publisher.validate_safe_plan_payload(
            plan,
            expected_day=date(2026, 8, 11),
            now=generated,
        )


def test_current_plan_uses_green_target_day_even_when_older_day_is_red(
    tmp_path: Path,
) -> None:
    db, manifest_path = _ready_fixture(tmp_path, analyzed=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    green = manifest["daily_verdicts"]["2026-08-11"]
    red = json.loads(json.dumps(green))
    red.update(
        day="2026-08-10",
        mango_unique=2,
        unexplained_missing=1,
        consistency_ok=False,
        closure_ok=False,
    )
    red_source = red["mango_enumeration_source"]
    red_source.update(
        since="2026-08-09T21:00:00+00:00",
        rolling_since="2026-08-09T21:00:00+00:00",
        until="2026-08-10T21:00:00+00:00",
    )
    red_source["covered_intervals"] = [
        {
            "since": "2026-08-09T21:00:00+00:00",
            "until": "2026-08-10T21:00:00+00:00",
            "result_complete": True,
        }
    ]
    red["mango_enumeration_source"] = dual_strict_source(
        red_source,
        call_keys=["call-1", "missing-call"],
        calls_by_day={"2026-08-10": ["call-1", "missing-call"]},
    )
    manifest.update(consistency_ok=False, closure_ok=False)
    manifest["moscow_dates"] = ["2026-08-10", "2026-08-11"]
    manifest["daily_verdicts"] = {
        "2026-08-10": red,
        "2026-08-11": green,
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    rows = publisher.load_manager_rows(
        ready_db=db,
        ready_manifest=manifest_path,
        ready_manifest_payload=manifest,
        day=date(2026, 8, 11),
        owner_email=OWNER,
        allowed_emails=(SERVICE, ROP),
    )
    plan = publisher.build_safe_plan(
        day=date(2026, 8, 11),
        rows=rows,
        ready_manifest=manifest,
        now=datetime(2026, 8, 11, 10, tzinfo=timezone.utc),
    )

    assert plan["consistency_ok"] is True
    assert plan["moscow_day"] == "2026-08-11"


def test_zero_stage10_rejects_nonempty_manager_rows(tmp_path: Path) -> None:
    _db, manifest_path = _ready_fixture(tmp_path, analyzed=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    verdict = manifest["daily_verdicts"]["2026-08-11"]
    verdict.update(
        mango_unique=0,
        ready_unique=0,
        quarantine_unique=0,
        pending_unique=0,
        unexplained_missing=0,
        independent_zero_enumerations=2,
        closure_ok=True,
    )
    original_source = manifest["mango_enumeration_source"]
    zero_source = dual_strict_source(
        {
            "mode": "strict_service",
            "since": original_source["since"],
            "rolling_since": original_source["rolling_since"],
            "until": original_source["until"],
            "cursor": original_source["cursor"],
            "pages": original_source["pages"],
            "pagination": original_source["pagination"],
            "requests": 1,
            "covered_intervals": [
                {
                    "since": original_source["rolling_since"],
                    "until": original_source["until"],
                    "result_complete": True,
                }
            ],
            "catch_up": False,
        },
        call_keys=[],
        calls_by_day={},
    )
    verdict["mango_enumeration_source"] = zero_source
    manifest["mango_enumeration_source"] = zero_source
    capture_proof, capture_proof_sha256 = ready_capture_proof(
        zero_source,
        zero_by_day={"2026-08-11": 2},
    )
    manifest["capture_proof"] = capture_proof
    manifest["capture_proof_sha256"] = capture_proof_sha256
    manifest["capture_proof_run_id"] = zero_source["dual_enumeration"][
        "proof_run_id"
    ]

    with pytest.raises(RuntimeError, match="Stage10 balance"):
        publisher.build_safe_plan(
            day=date(2026, 8, 11),
            rows=[manager_row("impossible-row")],
            ready_manifest=manifest,
            now=datetime(2026, 8, 11, 10, tzinfo=timezone.utc),
        )


def test_manifest_and_publication_lock_symlinks_are_rejected(
    tmp_path: Path,
) -> None:
    real_manifest = tmp_path / "real-manifest.json"
    real_manifest.write_text("{}\n", encoding="utf-8")
    manifest_link = tmp_path / "manifest.json"
    manifest_link.symlink_to(real_manifest)
    with pytest.raises(RuntimeError, match="manifest"):
        publisher.stable_json_object(manifest_link, label="ready manifest")

    lock_dir = tmp_path / "locks"
    lock_dir.mkdir(mode=0o700)
    victim = tmp_path / "victim.txt"
    victim.write_text("must remain unchanged", encoding="utf-8")
    lock = lock_dir / "publication.lock"
    lock.symlink_to(victim)
    with pytest.raises(RuntimeError, match="lock is unsafe"):
        with publisher.publication_lock(lock):
            pass
    assert victim.read_text(encoding="utf-8") == "must remain unchanged"

def test_dry_run_uses_one_ready_manifest_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db, manifest_path = _ready_fixture(tmp_path, analyzed=True)
    original = publisher.stable_json_object
    reads: list[object] = []
    seen_payloads: list[object] = []

    def stable(path: Path, *, label: str) -> dict[str, object]:
        reads.append((path, label))
        return dict(original(path, label=label))

    def rows(**kwargs: object) -> list[dict[str, object]]:
        seen_payloads.append(kwargs.get("ready_manifest_payload"))
        return [manager_row("call-1")]

    monkeypatch.setattr(publisher, "stable_json_object", stable)
    monkeypatch.setattr(publisher, "load_manager_rows", rows)

    assert publisher.main(
        [
            "--ready-db",
            str(db),
            "--ready-manifest",
            str(manifest_path),
            "--owner-email",
            OWNER,
            "--day",
            "2026-08-11",
        ]
    ) == 0
    assert len(reads) == 1
    assert len(seen_payloads) == 1
    assert isinstance(seen_payloads[0], dict)


def test_dry_run_recovers_interrupted_ready_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db, manifest_path = _ready_fixture(tmp_path, analyzed=True)
    _interrupt_ready_publication(db)
    monkeypatch.setattr(
        publisher,
        "load_manager_rows",
        lambda **_kwargs: [manager_row("call-1")],
    )

    result = publisher.main(
        [
            "--ready-db",
            str(db),
            "--ready-manifest",
            str(manifest_path),
            "--owner-email",
            OWNER,
            "--day",
            "2026-08-11",
        ]
    )

    assert result == 0
    assert inspect_ready_publication(db)["recovery_required"] is False


def test_execute_refuses_without_reads_network_or_state_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    touched: list[str] = []

    def forbidden(*_args: object, **_kwargs: object) -> object:
        touched.append("called")
        raise AssertionError("legacy execute touched an external or local writer path")

    for name in (
        "atomic_owner_json",
        "authorized_session",
        "owner_json",
        "stable_json_object",
    ):
        monkeypatch.setattr(publisher, name, forbidden)
    state = tmp_path / "state.json"

    with pytest.raises(
        RuntimeError,
        match=r"plan-only; use publish_live_mango_calls_google\.py",
    ):
        publisher.main(
            [
                "--execute",
                "--confirmation",
                "legacy-confirmation",
                "--config",
                "/not/read/config.json",
                "--safe-plan",
                "/not/read/plan.json",
                "--state",
                str(state),
                "--approved-plan-sha256",
                "0" * 64,
                "--day",
                "2026-08-11",
            ]
        )

    assert touched == []
    assert not state.exists()


def test_plan_only_module_has_no_google_post_or_uncertain_write_state() -> None:
    source = Path(publisher.__file__).read_text(encoding="utf-8")

    assert "session.post" not in source
    assert "write_uncertain" not in source
