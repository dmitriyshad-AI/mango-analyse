from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path

from mango_mvp.customer_timeline.mail_stage2_ingest import (
    MailStage2IngestConfig,
    apply_stage2_mail_ingest,
    create_timeline_backup,
    dry_run_stage2_mail_ingest,
    plan_stage2_mail_ingest,
    restore_timeline_backup,
)
from mango_mvp.customer_timeline.store import CustomerTimelineSQLiteStore
from mango_mvp.productization.mail_archive import TallantoIdentityMapConfig, build_tallanto_identity_map
from scripts.run_mail_stage2_timeline_ingest_procedure import main as ingest_cli_main


def _write_tallanto_csv(path: Path) -> None:
    rows = [
        {
            "ID": "T-1",
            "amoCRM ID": "A-1",
            "Имя": "Петр",
            "Фамилия": "Иванов",
            "ФИО родителя": "Мария Иванова",
            "E-mail": "parent@example.com",
            "Другой E-mail": "",
            "Тел. (родителя)": "+7 999 111-22-33",
        }
    ]
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )
    return path


def _make_config(tmp_path: Path) -> MailStage2IngestConfig:
    tallanto_csv = tmp_path / "tallanto.csv"
    _write_tallanto_csv(tallanto_csv)
    identity_report = build_tallanto_identity_map(
        TallantoIdentityMapConfig(
            tallanto_csv_path=tallanto_csv,
            out_dir=tmp_path / "identity",
            encoding="utf-8",
            delimiter="\t",
        )
    )
    assert identity_report["identity_values"]["email"]["strong_unique"] == 1
    extracted_root = tmp_path / "_external_handoffs" / "mail" / "extracted_text"
    extracted_root.mkdir(parents=True)
    linked_text = extracted_root / "linked.txt"
    linked_text.write_text("Клиент спрашивает про физику 8 класса и формат занятий.", encoding="utf-8")
    unmatched_text = extracted_root / "unmatched.txt"
    unmatched_text.write_text("Письмо без надёжной привязки к клиенту.", encoding="utf-8")
    events_path = _write_jsonl(
        tmp_path / "_external_handoffs" / "mail" / "stage2_events.jsonl",
        [
            {
                "message_sha256": "a" * 64,
                "customer_id": "interim:stale",
                "date_iso": "2026-06-20T10:00:00+00:00",
                "subject": "Физика 8 класс",
                "from": [{"email": "parent@example.com"}],
                "to": [{"email": "school@kmipt.ru"}],
                "brand": "foton",
                "brand_source": "mailbox",
                "extracted_text_path": str(linked_text),
            },
            {
                "message_sha256": "b" * 64,
                "date_iso": "2026-06-20T11:00:00+00:00",
                "subject": "Неизвестный клиент",
                "from": [{"email": "unknown@example.com"}],
                "to": [{"email": "school@kmipt.ru"}],
                "brand": "foton",
                "brand_source": "mailbox",
                "extracted_text_path": str(unmatched_text),
            },
        ],
    )
    db_path = tmp_path / "timeline" / "customer_timeline.sqlite"
    store = CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path)
    store.close()
    return MailStage2IngestConfig(
        timeline_db_path=db_path,
        allowed_root=tmp_path,
        identity_db_path=tmp_path / "identity" / "tallanto_email_identity_map.sqlite",
        event_jsonl_paths=(events_path,),
        out_dir=tmp_path / "reports",
        backup_root=tmp_path / "backups",
        tenant_id="foton",
        source_ref="test_fresh_relink_bacdd96f",
    )


def _count_rows(db_path: Path, table: str) -> int:
    with sqlite3.connect(db_path) as con:
        return int(con.execute(f"SELECT count(*) FROM {table}").fetchone()[0])


def test_mail_stage2_plan_uses_fresh_relink_and_does_not_trust_existing_customer_id(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    plans, counters = plan_stage2_mail_ingest(config)

    assert counters["input_events"] == 2
    assert counters["linked"] == 1
    assert counters["unmatched"] == 1
    linked = next(plan for plan in plans if plan.customer is not None)
    unmatched = next(plan for plan in plans if plan.customer is None)
    assert linked.customer.customer_id == "tallanto:T-1"
    assert linked.event.customer_id == "tallanto:T-1"
    assert "interim:stale" not in linked.event.customer_id
    assert linked.chunk is not None
    assert linked.chunk.allowed_for_bot is False
    assert linked.chunk.requires_manager_review is True
    assert unmatched.event.customer_id is None
    assert unmatched.event.metadata["pending_attribution"] is True
    assert unmatched.chunk is None


def test_mail_stage2_procedure_requires_backup_and_is_idempotent_then_restores(tmp_path: Path) -> None:
    config = _make_config(tmp_path)

    assert (
        ingest_cli_main(
            [
                "apply",
                "--timeline-db",
                str(config.timeline_db_path),
                "--allowed-root",
                str(config.allowed_root),
                "--identity-db",
                str(config.identity_db_path),
                "--event-jsonl",
                str(config.event_jsonl_paths[0]),
                "--out-dir",
                str(config.out_dir),
            ]
        )
        == 2
    )

    dry_run = dry_run_stage2_mail_ingest(config)
    assert dry_run["counts"]["would_create_events"] == 2
    assert _count_rows(config.timeline_db_path, "timeline_events") == 0

    backup = create_timeline_backup(config, label="test")
    backup_manifest = Path(str(backup["manifest_path"]))
    first = apply_stage2_mail_ingest(config, backup_manifest_path=backup_manifest)
    second = apply_stage2_mail_ingest(config, backup_manifest_path=backup_manifest)

    assert first["counts"]["created_events"] == 2
    assert first["counts"]["created_chunks"] == 1
    assert first["counts"]["pending_attribution_events"] == 1
    assert second["counts"]["selected_new_events"] == 0
    assert second["counts"]["created_events"] == 0
    assert second["counts"]["created_chunks"] == 0
    assert second["counts"]["skipped_existing_events"] == 2

    with sqlite3.connect(config.timeline_db_path) as con:
        chunks = con.execute(
            "SELECT allowed_for_bot, requires_manager_review FROM bot_context_chunks"
        ).fetchall()
        assert chunks == [(0, 1)]
        pending = con.execute(
            "SELECT customer_id, json_extract(record_json, '$.metadata.pending_attribution') FROM timeline_events "
            "WHERE match_status = 'unmatched'"
        ).fetchall()
        assert pending == [(None, 1)]

    restore = restore_timeline_backup(config, backup_manifest_path=backup_manifest)
    assert Path(str(restore["timeline_db_path"])) == config.timeline_db_path
    assert _count_rows(config.timeline_db_path, "timeline_events") == 0
    assert _count_rows(config.timeline_db_path, "bot_context_chunks") == 0
