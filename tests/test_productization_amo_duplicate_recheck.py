from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

from mango_mvp.productization.amo_duplicate_recheck import build_amo_duplicate_post_merge_recheck


def test_duplicate_recheck_pending_when_dry_run_missing(tmp_path: Path) -> None:
    pack = _fixture_duplicate_pack(tmp_path)
    out = tmp_path / "out"

    summary = build_amo_duplicate_post_merge_recheck(duplicate_pack_root=pack, reports_root=tmp_path / "reports", out_root=out)

    assert summary["status"] == "pending_not_run"
    assert summary["passed"] is False
    assert summary["policy"]["write_crm"] is False
    rows = list(csv.DictReader((out / "row_results.csv").open(encoding="utf-8-sig")))
    assert rows[0]["blocking_reason"] == "missing_phone_in_dry_run_report"


def test_duplicate_recheck_passes_one_surviving_contact_per_phone(tmp_path: Path) -> None:
    pack = _fixture_duplicate_pack(tmp_path)
    report = _fixture_report(tmp_path, input_csv=pack / "post_merge_recheck_input_ru.csv", rows=[_report_row("79000000001", "dry_run", "live_write_not_confirmed", "111")])
    out = tmp_path / "out"

    summary = build_amo_duplicate_post_merge_recheck(duplicate_pack_root=pack, report_dir=report, out_root=out)

    assert summary["status"] == "passed"
    assert summary["passed"] is True
    assert summary["ready_after_merge_rows"] == 1
    row = list(csv.DictReader((out / "row_results.csv").open(encoding="utf-8-sig")))[0]
    assert row["decision"] == "ready_after_merge"
    assert row["surviving_contact_id"] == "111"


def test_duplicate_recheck_blocks_multiple_or_mismatch(tmp_path: Path) -> None:
    pack = _fixture_duplicate_pack(tmp_path)
    report = _fixture_report(tmp_path, input_csv=pack / "post_merge_recheck_input_ru.csv", rows=[_report_row("79000000001", "skipped", "multiple_exact_contacts_in_amo", "111 | 112")])

    summary = build_amo_duplicate_post_merge_recheck(duplicate_pack_root=pack, report_dir=report, out_root=tmp_path / "out")

    assert summary["status"] == "blocked"
    assert summary["passed"] is False
    assert summary["blocking_counts"] == {"multiple_exact_contacts_in_amo": 1}


def test_duplicate_recheck_allows_known_candidate_survivor_outside_old_source(tmp_path: Path) -> None:
    pack = _fixture_duplicate_pack(tmp_path)
    report = _fixture_report(tmp_path, input_csv=pack / "post_merge_recheck_input_ru.csv", rows=[_report_row("79000000001", "dry_run", "live_write_not_confirmed", "112")])

    summary = build_amo_duplicate_post_merge_recheck(duplicate_pack_root=pack, report_dir=report, out_root=tmp_path / "out")

    assert summary["status"] == "passed"
    row = list(csv.DictReader((tmp_path / "out" / "row_results.csv").open(encoding="utf-8-sig")))[0]
    assert row["decision"] == "ready_after_merge"
    assert row["survivor_relation"] == "known_candidate_outside_source"


def test_duplicate_recheck_rejects_stale_explicit_report_dir(tmp_path: Path) -> None:
    pack = _fixture_duplicate_pack(tmp_path)
    stale_input = tmp_path / "other_input.csv"
    stale_input.write_text("Телефон клиента\n79000000001\n", encoding="utf-8")
    report = _fixture_report(tmp_path, input_csv=stale_input, rows=[_report_row("79000000001", "dry_run", "live_write_not_confirmed", "111")])

    summary = build_amo_duplicate_post_merge_recheck(duplicate_pack_root=pack, report_dir=report, out_root=tmp_path / "out")

    assert summary["status"] == "blocked"
    assert "dry_run_summary_input_mismatch" in summary["global_blockers"]


def test_duplicate_recheck_cli_returns_nonzero_for_pending(tmp_path: Path) -> None:
    pack = _fixture_duplicate_pack(tmp_path)
    out = tmp_path / "out"
    completed = subprocess.run(
        [
            "python3",
            "scripts/check_amo_duplicate_post_merge_recheck.py",
            "--duplicate-pack-root",
            str(pack),
            "--reports-root",
            str(tmp_path / "reports"),
            "--out-root",
            str(out),
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 1
    assert "pending_not_run" in completed.stdout


def _fixture_duplicate_pack(tmp_path: Path) -> Path:
    pack = tmp_path / "duplicate_pack"
    pack.mkdir()
    _write_csv(
        pack / "duplicate_merge_queue.csv",
        [
            {
                "resolution_id": "row0001_79000000001",
                "phone": "79000000001",
                "source_amo_contact_ids": "111",
                "dry_run_contact_ids": "111 | 112",
                "duplicate_resolution_status": "duplicate_contacts_merge_required",
            }
        ],
    )
    _write_csv(
        pack / "candidate_contacts.csv",
        [
            {"resolution_id": "row0001_79000000001", "phone": "79000000001", "candidate_contact_id": "111"},
            {"resolution_id": "row0001_79000000001", "phone": "79000000001", "candidate_contact_id": "112"},
        ],
    )
    _write_csv(
        pack / "post_merge_recheck_input_ru.csv",
        [
            {
                "Телефон клиента": "79000000001",
                "AMO contact IDs": "111",
                "Manual resolution id": "row0001_79000000001",
                "Duplicate resolution status": "duplicate_contacts_merge_required",
            }
        ],
    )
    return pack


def _fixture_report(tmp_path: Path, *, input_csv: Path, rows: list[dict[str, str]]) -> Path:
    report = tmp_path / "reports" / "20260511T000000Z"
    report.mkdir(parents=True)
    summary = {
        "run_id": "20260511T000000Z",
        "mode": "dry_run",
        "live_write": False,
        "input": str(input_csv.resolve(strict=False)),
        "total_rows": len(rows),
        "dry_run": sum(1 for row in rows if row["status"] == "dry_run"),
        "skipped": sum(1 for row in rows if row["status"] == "skipped"),
        "failed": sum(1 for row in rows if row["status"] == "failed"),
    }
    (report / "contact_writeback_summary.json").write_text(json.dumps(summary, ensure_ascii=False), encoding="utf-8")
    _write_csv(report / "contact_writeback_report.csv", rows)
    return report


def _report_row(phone: str, status: str, reason: str, contact_id: str) -> dict[str, str]:
    return {
        "row_index": "1",
        "mode": "dry_run",
        "phone": phone,
        "status": status,
        "reason": reason,
        "contact_id": contact_id,
        "contact_name": "Test",
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
