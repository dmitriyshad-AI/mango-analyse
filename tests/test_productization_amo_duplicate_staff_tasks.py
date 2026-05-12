from __future__ import annotations

import csv
import subprocess
from pathlib import Path

from mango_mvp.productization.amo_duplicate_staff_tasks import build_amo_duplicate_staff_tasks


def test_duplicate_staff_tasks_are_read_only_and_actionable(tmp_path: Path) -> None:
    pack = _fixture_duplicate_pack(tmp_path)
    out = tmp_path / "tasks"

    summary = build_amo_duplicate_staff_tasks(duplicate_pack_root=pack, out_root=out)

    assert summary["task_rows"] == 2
    assert summary["policy"]["write_amo"] is False
    assert summary["policy"]["post_merge_recheck_required"] is True
    tasks = list(csv.DictReader((out / "staff_tasks.csv").open(encoding="utf-8-sig")))
    assert tasks[0]["post_merge_recheck_required"] == "yes"
    assert "объединить дубли" in tasks[0]["instruction_ru"]
    assert "проверить где реальная карточка" in tasks[1]["instruction_ru"]
    manager_rows = list(csv.DictReader((out / "manager_summary.csv").open(encoding="utf-8-sig")))
    assert manager_rows
    assert (out / "staff_tasks.html").read_text(encoding="utf-8").count("AMO duplicate staff tasks") >= 1


def test_duplicate_staff_tasks_cli(tmp_path: Path) -> None:
    pack = _fixture_duplicate_pack(tmp_path)
    out = tmp_path / "tasks"
    completed = subprocess.run(
        [
            "python3",
            "scripts/build_amo_duplicate_staff_tasks.py",
            "--duplicate-pack-root",
            str(pack),
            "--out-root",
            str(out),
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0
    assert "amo_duplicate_staff_tasks_v1" in completed.stdout
    assert (out / "staff_tasks.csv").exists()


def _fixture_duplicate_pack(tmp_path: Path) -> Path:
    pack = tmp_path / "duplicate_pack"
    pack.mkdir()
    _write_csv(
        pack / "duplicate_merge_queue.csv",
        [
            {
                "resolution_id": "row1_79000000001",
                "phone": "79000000001",
                "duplicate_resolution_status": "duplicate_contacts_merge_required",
                "owner_hint": "manager_who_owns_client_context",
                "last_call_manager": "Менеджер 1",
                "source_amo_contact_ids": "111",
                "dry_run_contact_ids": "111 | 112",
                "suggested_keep_contact_id": "111",
                "all_candidate_contact_ids": "111 | 112",
                "contact_links": "https://example.test/111\nhttps://example.test/112",
                "lead_links": "https://example.test/lead/1",
                "fio_child": "Ученик",
                "merge_priority": "high",
                "sale_probability_percent": "70",
                "next_step": "Отправить ссылку на оплату",
            },
            {
                "resolution_id": "row2_79000000002",
                "phone": "79000000002",
                "duplicate_resolution_status": "contact_id_mismatch_requires_operator",
                "owner_hint": "amo_operator_or_last_call_manager",
                "last_call_manager": "Менеджер 2",
                "source_amo_contact_ids": "221",
                "dry_run_contact_ids": "222",
                "all_candidate_contact_ids": "221 | 222",
                "contact_links": "https://example.test/221\nhttps://example.test/222",
                "merge_priority": "medium",
            },
        ],
    )
    _write_csv(
        pack / "candidate_contacts.csv",
        [
            {"resolution_id": "row1_79000000001", "phone": "79000000001", "candidate_contact_id": "111"},
            {"resolution_id": "row1_79000000001", "phone": "79000000001", "candidate_contact_id": "112"},
            {"resolution_id": "row2_79000000002", "phone": "79000000002", "candidate_contact_id": "221"},
            {"resolution_id": "row2_79000000002", "phone": "79000000002", "candidate_contact_id": "222"},
        ],
    )
    return pack


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
