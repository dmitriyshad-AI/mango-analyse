from __future__ import annotations

import csv
import subprocess
from pathlib import Path

from mango_mvp.productization.amo_manual_resolution import build_amo_manual_resolution_pack


def test_manual_resolution_pack_is_fail_closed_by_default(tmp_path: Path) -> None:
    queue_root, source = _fixture_queue(tmp_path)
    out = tmp_path / "manual_resolution"

    result = build_amo_manual_resolution_pack(queue_root=queue_root, source_csv=source, out_root=out)

    assert result["summary"]["review_rows"] == 4
    assert result["summary"]["accepted_rows"] == 0
    assert result["summary"]["resolved_live_candidate_rows"] == 0
    assert result["summary"]["needs_human_rows"] == 3
    assert result["summary"]["already_written_review_rows"] == 1
    assert result["summary"]["still_blocked_rows"] == 0
    assert (out / "resolution_template.csv").exists()
    assert (out / "resolved_live_candidates_ru.csv").read_text(encoding="utf-8-sig").strip().startswith("Телефон клиента")
    assert "No resolved live candidates" in (out / "next_dry_run_command.sh").read_text(encoding="utf-8")
    dry_run = subprocess.run([str(out / "next_dry_run_command.sh")], cwd=Path.cwd(), text=True, capture_output=True, check=False)
    assert dry_run.returncode == 0
    assert "No resolved live candidates" in dry_run.stdout


def test_manual_resolution_pack_applies_accepted_source_contact_id(tmp_path: Path) -> None:
    queue_root, source = _fixture_queue(tmp_path)
    first = build_amo_manual_resolution_pack(queue_root=queue_root, source_csv=source, out_root=tmp_path / "first")
    template_rows = list(csv.DictReader(Path(first["outputs"]["resolution_template_csv"]).open(encoding="utf-8-sig")))
    decision = next(row for row in template_rows if row["phone"] == "+79000000001")
    decision["resolution_status"] = "accepted_by_operator"
    decision["resolved_contact_id"] = "111"
    decision["resolution_reason"] = "operator_confirmed_source_contact_post_merge_recheck_approved"
    decision["resolved_by"] = "test_operator"
    decision_path = tmp_path / "decisions.csv"
    _write_csv(decision_path, [decision])

    result = build_amo_manual_resolution_pack(
        queue_root=queue_root,
        source_csv=source,
        out_root=tmp_path / "second",
        decisions_csv=decision_path,
    )

    assert result["summary"]["accepted_rows"] == 1
    assert result["summary"]["resolved_live_candidate_rows"] == 1
    rows = list(csv.DictReader(Path(result["outputs"]["resolved_live_candidates_csv"]).open(encoding="utf-8-sig")))
    assert rows[0]["Телефон клиента"] == "+79000000001"
    assert rows[0]["AMO contact IDs"] == "111"
    assert rows[0]["Manual resolution status"] == "accepted_by_operator"


def test_manual_resolution_blocks_accepted_contact_outside_source_without_override(tmp_path: Path) -> None:
    queue_root, source = _fixture_queue(tmp_path)
    first = build_amo_manual_resolution_pack(queue_root=queue_root, source_csv=source, out_root=tmp_path / "first")
    template_rows = list(csv.DictReader(Path(first["outputs"]["resolution_template_csv"]).open(encoding="utf-8-sig")))
    decision = next(row for row in template_rows if row["phone"] == "+79000000002")
    decision["resolution_status"] = "accepted_by_operator"
    decision["resolved_contact_id"] = "999"
    decision["resolution_reason"] = "operator_confirmed_source_contact_post_merge_recheck_approved"
    decision["resolved_by"] = "test_operator"
    decision_path = tmp_path / "bad_decisions.csv"
    _write_csv(decision_path, [decision])

    result = build_amo_manual_resolution_pack(
        queue_root=queue_root,
        source_csv=source,
        out_root=tmp_path / "second",
        decisions_csv=decision_path,
    )

    assert result["summary"]["accepted_rows"] == 1
    assert result["summary"]["resolved_live_candidate_rows"] == 0
    assert result["summary"]["blocked"] == 1
    blocked = list(csv.DictReader(Path(result["outputs"]["still_blocked_csv"]).open(encoding="utf-8-sig")))
    assert blocked[0]["validation_error"] == "resolved_contact_id_not_in_source_amo_contact_ids"


def test_manual_resolution_blocks_multi_contact_acceptance_without_post_merge_recheck(tmp_path: Path) -> None:
    queue_root, source = _fixture_queue(tmp_path)
    first = build_amo_manual_resolution_pack(queue_root=queue_root, source_csv=source, out_root=tmp_path / "first")
    template_rows = list(csv.DictReader(Path(first["outputs"]["resolution_template_csv"]).open(encoding="utf-8-sig")))
    decision = next(row for row in template_rows if row["phone"] == "+79000000001")
    decision["resolution_status"] = "accepted_by_operator"
    decision["resolved_contact_id"] = "111"
    decision["resolution_reason"] = "operator_confirmed_source_contact"
    decision["resolved_by"] = "test_operator"
    decision_path = tmp_path / "bad_multi_decisions.csv"
    _write_csv(decision_path, [decision])

    result = build_amo_manual_resolution_pack(
        queue_root=queue_root,
        source_csv=source,
        out_root=tmp_path / "second",
        decisions_csv=decision_path,
    )

    assert result["summary"]["resolved_live_candidate_rows"] == 0
    blocked = list(csv.DictReader(Path(result["outputs"]["still_blocked_csv"]).open(encoding="utf-8-sig")))
    assert blocked[0]["validation_error"] == "duplicate_merge_requires_post_merge_recheck_approved_reason"


def test_manual_resolution_blocks_accepted_decision_without_reason_and_resolved_by(tmp_path: Path) -> None:
    queue_root, source = _fixture_queue(tmp_path)
    first = build_amo_manual_resolution_pack(queue_root=queue_root, source_csv=source, out_root=tmp_path / "first")
    template_rows = list(csv.DictReader(Path(first["outputs"]["resolution_template_csv"]).open(encoding="utf-8-sig")))
    decision = next(row for row in template_rows if row["phone"] == "+79000000001")
    decision["resolution_status"] = "accepted_by_operator"
    decision["resolved_contact_id"] = "111"
    decision_path = tmp_path / "bad_decisions.csv"
    _write_csv(decision_path, [decision])

    result = build_amo_manual_resolution_pack(
        queue_root=queue_root,
        source_csv=source,
        out_root=tmp_path / "second",
        decisions_csv=decision_path,
    )

    assert result["summary"]["resolved_live_candidate_rows"] == 0
    blocked = list(csv.DictReader(Path(result["outputs"]["still_blocked_csv"]).open(encoding="utf-8-sig")))
    assert blocked[0]["validation_error"] == "accepted_resolution_requires_reason"


def test_manual_resolution_blocks_already_written_text_quality_refresh_without_explicit_policy(tmp_path: Path) -> None:
    queue_root, source = _fixture_queue(tmp_path)
    first = build_amo_manual_resolution_pack(queue_root=queue_root, source_csv=source, out_root=tmp_path / "first")
    template_rows = list(csv.DictReader(Path(first["outputs"]["resolution_template_csv"]).open(encoding="utf-8-sig")))
    decision = next(row for row in template_rows if row["phone"] == "+79000000004")
    decision["resolution_status"] = "accepted_by_operator"
    decision["resolved_contact_id"] = "444"
    decision["resolution_reason"] = "text_quality_approved"
    decision["resolved_by"] = "test_operator"
    decision_path = tmp_path / "bad_refresh.csv"
    _write_csv(decision_path, [decision])

    result = build_amo_manual_resolution_pack(
        queue_root=queue_root,
        source_csv=source,
        out_root=tmp_path / "second",
        decisions_csv=decision_path,
    )

    assert result["summary"]["resolved_live_candidate_rows"] == 0
    blocked = list(csv.DictReader(Path(result["outputs"]["still_blocked_csv"]).open(encoding="utf-8-sig")))
    assert blocked[0]["validation_error"] == "already_written_review_requires_refresh_approved_reason"


def _fixture_queue(tmp_path: Path) -> tuple[Path, Path]:
    queue_root = tmp_path / "queue"
    queue_root.mkdir()
    source = tmp_path / "amo_export_ready_ru.csv"
    _write_csv(
        source,
        [
            _source_row("+79000000001", "111"),
            _source_row("+79000000002", "222"),
            _source_row("+79000000003", "333"),
            _source_row("+79000000004", "444"),
        ],
    )
    _write_csv(
        queue_root / "needs_manager_review_multi_contact.csv",
        [
            _queue_row("needs_manager_review_multi_contact", "+79000000001", "111", "111 | 112", "1"),
            _queue_row("needs_manager_review_multi_contact", "+79000000002", "222", "222 | 223", "2"),
        ],
    )
    _write_csv(
        queue_root / "blocked_contact_id_mismatch.csv",
        [_queue_row("blocked_contact_id_mismatch", "+79000000003", "333", "334", "3")],
    )
    text_row = _queue_row("needs_text_quality_review", "+79000000004", "444", "", "4")
    text_row["written_status"] = "written"
    text_row["written_contact_id"] = "444"
    _write_csv(queue_root / "needs_text_quality_review.csv", [text_row])
    return queue_root, source


def _source_row(phone: str, amo_id: str) -> dict[str, str]:
    return {
        "Телефон клиента": phone,
        "AMO contact IDs": amo_id,
        "AMO lead IDs": "lead",
        "Готово к записи в AMO": "Да",
        "Тип последнего свежего звонка": "sales_call",
        "Следующий шаг": "Позвонить",
        "Приоритет лида": "warm",
    }


def _queue_row(bucket: str, phone: str, source_ids: str, dry_ids: str, index: str) -> dict[str, str]:
    return {
        "queue_bucket": bucket,
        "queue_reason": bucket,
        "source_row_index": index,
        "normalized_phone": phone,
        "source_amo_contact_ids": source_ids,
        "effective_contact_id": source_ids,
        "dry_run_status": "skipped",
        "dry_run_reason": bucket,
        "dry_run_contact_id": dry_ids,
        "written_status": "",
        "written_contact_id": "",
        "Дата последнего свежего звонка": "2026-04-01",
        "Тип последнего свежего звонка": "sales_call",
        "Приоритет лида": "warm",
        "Вероятность продажи, %": "65",
        "Следующий шаг": "Позвонить",
        "ФИО родителя": "",
        "ФИО ребенка": "",
        "AMO lead IDs": "lead",
    }


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
