from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

from mango_mvp.productization.amo_duplicate_resolution import build_amo_duplicate_resolution_pack


def test_amo_duplicate_resolution_pack_is_read_only_and_fail_closed(tmp_path: Path) -> None:
    manual_pack = _fixture_manual_pack(tmp_path)
    runtime = _fixture_runtime(tmp_path)
    out = tmp_path / "duplicate_resolution"

    result = build_amo_duplicate_resolution_pack(
        manual_pack_root=manual_pack,
        out_root=out,
        current_runtime_path=runtime,
        generated_at=None,
    )

    assert result["review_rows"] == 2
    assert result["candidate_contact_rows"] == 4
    assert result["policy"]["write_crm"] is False
    assert result["policy"]["live_write_executed"] is False
    assert result["by_duplicate_resolution_status"] == {
        "contact_id_mismatch_requires_operator": 1,
        "duplicate_contacts_merge_required": 1,
    }
    queue_rows = list(csv.DictReader((out / "duplicate_merge_queue.csv").open(encoding="utf-8-sig")))
    assert queue_rows[0]["duplicate_resolution_status"] == "duplicate_contacts_merge_required"
    assert "https://educent.amocrm.ru/contacts/detail/111" in queue_rows[0]["contact_links"]
    assert (out / "duplicate_merge_review.html").read_text(encoding="utf-8").count("AMO duplicate resolution") >= 1
    assert (out / "duplicate_merge_review.xlsx").exists()
    command = (out / "next_recheck_command.sh").read_text(encoding="utf-8")
    assert "--live-write" not in command
    assert "--expected-dry-run 2" in command


def test_amo_duplicate_resolution_recheck_command_is_dry_run_only(tmp_path: Path) -> None:
    manual_pack = _fixture_manual_pack(tmp_path, rows=[])
    runtime = _fixture_runtime(tmp_path)
    out = tmp_path / "duplicate_resolution"

    build_amo_duplicate_resolution_pack(manual_pack_root=manual_pack, out_root=out, current_runtime_path=runtime)

    command = out / "next_recheck_command.sh"
    completed = subprocess.run([str(command)], text=True, capture_output=True, check=False)
    assert completed.returncode == 0
    assert "No duplicate/contact-mismatch rows" in completed.stdout


def _fixture_manual_pack(tmp_path: Path, rows: list[dict[str, str]] | None = None) -> Path:
    root = tmp_path / "manual"
    root.mkdir()
    if rows is None:
        rows = [
            _row("row1", "needs_manager_review_multi_contact", "+79000000001", "111", "111 | 112", "lead1"),
            _row("row2", "blocked_contact_id_mismatch", "+79000000002", "222", "223", "lead2"),
            _row("row3", "needs_text_quality_review", "+79000000003", "333", "", "lead3"),
        ]
    _write_csv(root / "resolution_template.csv", rows)
    return root


def _fixture_runtime(tmp_path: Path) -> Path:
    root = tmp_path / "stable_runtime"
    root.mkdir()
    path = root / "CURRENT_RUNTIME.json"
    path.write_text(
        json.dumps(
            {
                "paths": {
                    "stage15_summary": str(tmp_path / "stage15.json"),
                    "crm_quality_summary": str(tmp_path / "crm_quality.json"),
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return path


def _row(resolution_id: str, bucket: str, phone: str, source_ids: str, dry_ids: str, lead_ids: str) -> dict[str, str]:
    return {
        "resolution_id": resolution_id,
        "queue_bucket": bucket,
        "phone": phone,
        "source_amo_contact_ids": source_ids,
        "dry_run_contact_ids": dry_ids,
        "suggested_resolved_contact_id": source_ids.split("|")[0].strip(),
        "amo_lead_ids": lead_ids,
        "latest_call_date": "2026-04-01",
        "latest_call_type": "sales_call",
        "priority": "warm",
        "sale_probability_percent": "65",
        "next_step": "Отправить материалы",
        "fio_parent": "Иванова",
        "fio_child": "Петр",
    }


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
