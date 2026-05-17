from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.build_kc_night_audit_pack import REQUIRED_AUDIT_FILES, build_kc_night_audit_pack


def test_build_kc_night_audit_pack_writes_required_reports_without_raw_private_data(tmp_path: Path) -> None:
    snapshot_root = tmp_path / "snapshot"
    snapshot_root.mkdir()
    _write_fixture_snapshot(snapshot_root)

    pack_dir = tmp_path / "audits" / "telegram_pilot_kb_night_build_20260517_placeholder"
    result = build_kc_night_audit_pack(
        snapshot_root,
        pack_dir,
        changed_files=("scripts/build_kc_night_audit_pack.py", "tests/test_kc_night_audit_pack.py"),
        test_output="tests/test_kc_night_audit_pack.py::test_contract PASSED\nclient phone +79991234567",
    )

    assert result == pack_dir
    for filename in REQUIRED_AUDIT_FILES:
        assert (pack_dir / filename).exists(), filename

    all_report_text = "\n".join((pack_dir / filename).read_text(encoding="utf-8") for filename in REQUIRED_AUDIT_FILES)
    assert "всего источников: `3`" in all_report_text
    assert "всего фактов: `4`" in all_report_text
    assert "precise facts without fresh status: `1`" in all_report_text
    assert "rows in before/after comparison: `2`" in all_report_text
    assert "Stage 6 contains" not in all_report_text

    assert "client@example.com" not in all_report_text
    assert "+79991234567" not in all_report_text
    assert "Сырой вопрос клиента с персональными данными" not in all_report_text
    assert "Исторический ответ менеджера с телефоном" not in all_report_text
    assert "draft text with raw client content" not in all_report_text
    assert "[redacted_phone]" in (pack_dir / "test_output.txt").read_text(encoding="utf-8")


def test_build_kc_night_audit_pack_handles_missing_snapshot_files(tmp_path: Path) -> None:
    snapshot_root = tmp_path / "empty_snapshot"
    snapshot_root.mkdir()
    pack_dir = tmp_path / "audits" / "telegram_pilot_kb_night_build_20260517_empty"

    build_kc_night_audit_pack(snapshot_root, pack_dir)

    for filename in REQUIRED_AUDIT_FILES:
        assert (pack_dir / filename).exists(), filename

    risk_review = (pack_dir / "risk_review.md").read_text(encoding="utf-8")
    no_live = (pack_dir / "no_live_write_proof.md").read_text(encoding="utf-8")
    assert "source_inventory не найден или пустой" in risk_review
    assert "facts не найдены или пустые" in risk_review
    assert "Snapshot file was not found" in no_live


def test_build_kc_night_audit_pack_rejects_stable_runtime_output(tmp_path: Path) -> None:
    snapshot_root = tmp_path / "snapshot"
    snapshot_root.mkdir()

    with pytest.raises(ValueError, match="stable_runtime"):
        build_kc_night_audit_pack(snapshot_root, tmp_path / "stable_runtime" / "audit_pack")


def _write_fixture_snapshot(root: Path) -> None:
    _write_csv(
        root / "source_inventory.csv",
        [
            {
                "source_id": "doc_kc",
                "source_kind": "local_docx",
                "read_ok": "true",
                "freshness_status": "fresh_verified",
                "title": "База знаний КЦ",
                "path": "/private/client@example.com",
            },
            {
                "source_id": "drive_price_unpk",
                "source_kind": "google_drive_doc",
                "read_ok": "false",
                "freshness_status": "metadata_only",
                "title": "УНПК цены",
                "path": "https://drive.example/private",
            },
            {
                "source_id": "catalog",
                "source_kind": "question_catalog",
                "read_ok": "true",
                "freshness_status": "needs_manager_confirmation",
                "title": "question catalog",
                "path": "product_data/question_catalog/customer_question_items.jsonl",
            },
        ],
    )
    _write_jsonl(
        root / "facts.jsonl",
        [
            {
                "fact_id": "fact_price_ok",
                "fact_type": "price",
                "freshness_status": "fresh_verified",
                "usable_for_precise_answer": True,
                "client_safe_text": "Стоимость подтверждена. client@example.com",
            },
            {
                "fact_id": "fact_price_bad",
                "fact_type": "price",
                "freshness_status": "needs_manager_confirmation",
                "usable_for_precise_answer": True,
                "client_safe_text": "Нужно уточнить цену по телефону +79991234567",
            },
            {
                "fact_id": "fact_internal",
                "fact_type": "manager_instruction",
                "freshness_status": "internal_only",
                "forbidden_for_client": True,
                "manager_text": "Не показывать клиенту.",
            },
            {
                "fact_id": "fact_do_not_use",
                "fact_type": "schedule",
                "freshness_status": "do_not_use",
                "requires_manager_confirmation": True,
                "short_fact": "Расписание устарело.",
            },
        ],
    )
    _write_jsonl(
        root / "knowledge_chunks.jsonl",
        [
            {
                "chunk_id": "chunk_price",
                "source_id": "doc_kc",
                "freshness_status": "fresh_verified",
                "text": "Короткий chunk",
                "prompt_safe": True,
            },
            {
                "chunk_id": "chunk_long",
                "source_id": "drive_price_unpk",
                "freshness_status": "metadata_only",
                "text": "x" * 701,
                "prompt_safe": True,
            },
            {
                "chunk_id": "chunk_private",
                "source_id": "catalog",
                "freshness_status": "needs_manager_confirmation",
                "text": "Сырой вопрос клиента с персональными данными +79991234567",
                "contains_pii": True,
            },
        ],
    )
    _write_csv(
        root / "manager_answer_sample_300_500.csv",
        [
            {
                "channel": "telegram",
                "theme_id": "price",
                "question": "Сырой вопрос клиента с персональными данными",
                "manager_answer": "Исторический ответ менеджера с телефоном +79991234567",
            },
            {
                "channel": "email",
                "theme_id": "schedule",
                "question": "raw question",
                "manager_answer": "raw answer",
            },
        ],
    )
    _write_jsonl(
        root / "manager_answer_patterns.jsonl",
        [
            {"pattern_id": "p1", "risk_level": "safe", "usable_as_example": True, "example": "do not copy raw"},
            {"pattern_id": "p2", "risk_level": "commercial", "usable_as_example": False, "unsafe": True},
        ],
    )
    _write_csv(
        root / "unsafe_or_outdated_manager_answers.csv",
        [{"row_id": "1", "reason": "outdated price", "manager_answer": "raw private answer"}],
    )
    _write_csv(
        root / "stage6_before_after_comparison.csv",
        [
            {
                "dialog_id": "1",
                "became_more_substantive": "true",
                "used_kb_context": "true",
                "draft_text": "draft text with raw client content",
            },
            {
                "dialog_id": "2",
                "became_more_substantive": "false",
                "unsupported_numeric_promises": "1",
                "requires_manual_review": "true",
            },
        ],
    )
    _write_csv(
        root / "stage6_kb_enriched_drafts.csv",
        [
            {"dialog_id": "1", "route": "draft_for_manager", "topic_valid": "true", "kb_context_used": "true"},
            {"dialog_id": "2", "route": "manager_only", "invalid_topic": "true", "llm_error": "true"},
        ],
    )
    (root / "stage6_before_after_summary.md").write_text("summary with no raw copied", encoding="utf-8")
    (root / "quality_summary.json").write_text(json.dumps({"context_found_rate": 0.75}), encoding="utf-8")
    (root / "kc_snapshot_20260517_night_v1.json").write_text(
        json.dumps(
            {
                "safety": {
                    "google_drive_write": False,
                    "crm_write": False,
                    "tallanto_write": False,
                    "client_send": False,
                    "stable_runtime_write": False,
                    "send_full_docx_to_prompt": False,
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
