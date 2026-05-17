from __future__ import annotations

import csv
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from scripts.build_kc_final_release import build_kc_final_release


def test_final_release_blocks_precise_facts_and_preserves_read_only_safety(tmp_path: Path) -> None:
    google_exports = tmp_path / "gdocs"
    google_exports.mkdir()
    (google_exports / "foton_prices_2026_2027.txt").write_text(
        "ФОТОН цены 2026/2027\nСтоимость 49 000 руб.\nСкидка 10% до 15 мая.\n",
        encoding="utf-8",
    )
    (google_exports / "whatsapp_tg_quick_replies.txt").write_text(
        "Вопрос: Когда занятия?\nОтвет: Менеджер уточнит актуальное расписание и вернется с проверенным вариантом.\n",
        encoding="utf-8",
    )
    manager_patterns = tmp_path / "patterns.jsonl"
    manager_patterns.write_text(
        json.dumps(
            {
                "pattern_id": "p1",
                "safe_pattern_template": "Сначала признаем вопрос клиента, затем задаем уточняющий вопрос.",
                "usable_as_fact": True,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    result = build_kc_final_release(
        run_id="test_release",
        out_dir=tmp_path / "out",
        google_export_dir=google_exports,
        old_extract_dir=tmp_path / "missing_old",
        kc_docx_path=tmp_path / "missing.docx",
        manager_patterns_path=manager_patterns,
        manager_sample_path=tmp_path / "missing_sample.jsonl",
        structured_facts_path=tmp_path / "missing_facts.jsonl",
        answer_templates_path=tmp_path / "missing_templates.csv",
        approved_answers_draft_path=tmp_path / "missing_approved.csv",
        question_items_path=tmp_path / "missing_questions.jsonl",
        fact_source_registry_path=tmp_path / "missing_registry.json",
        include_old_extract=False,
        crawl_current_sites=False,
        site_page_limit=0,
    )

    snapshot = json.loads(Path(result["snapshot_path"]).read_text(encoding="utf-8"))
    assert snapshot["schema_version"] == "kc_knowledge_snapshot_v1"
    assert snapshot["safety"]["client_send"] is False
    assert snapshot["safety"]["stable_runtime_write"] is False
    assert snapshot["summary"]["usable_for_precise_answer"] == 0
    assert snapshot["facts"]
    assert all(fact["usable_for_precise_answer"] is False for fact in snapshot["facts"])
    assert all(fact["bot_permission"] == "manager_only" for fact in snapshot["facts"])
    assert snapshot["manager_answer_patterns"]
    assert all(pattern["usable_as_fact"] is False for pattern in snapshot["manager_answer_patterns"])


def test_final_release_copies_old_extracts_into_release_paths(tmp_path: Path) -> None:
    google_exports = tmp_path / "gdocs"
    old_extract = tmp_path / ".codex_local" / "kc_source_extract_20260513"
    site_texts = old_extract / "site_texts"
    google_exports.mkdir(parents=True)
    site_texts.mkdir(parents=True)
    (google_exports / "kc_knowledge_base.txt").write_text("База знаний КЦ\nОбщее правило ответа клиенту.\n", encoding="utf-8")
    (site_texts / "cdpofoton.ru_005_courses.txt").write_text(
        "Курсы ФОТОН\nПодготовка к ЕГЭ, ОГЭ, математика, физика, информатика.\n",
        encoding="utf-8",
    )

    result = build_kc_final_release(
        run_id="test_paths",
        out_dir=tmp_path / "out",
        google_export_dir=google_exports,
        old_extract_dir=old_extract,
        kc_docx_path=tmp_path / "missing.docx",
        manager_patterns_path=tmp_path / "missing_patterns.jsonl",
        manager_sample_path=tmp_path / "missing_sample.jsonl",
        structured_facts_path=tmp_path / "missing_facts.jsonl",
        answer_templates_path=tmp_path / "missing_templates.csv",
        approved_answers_draft_path=tmp_path / "missing_approved.csv",
        question_items_path=tmp_path / "missing_questions.jsonl",
        fact_source_registry_path=tmp_path / "missing_registry.json",
        include_old_extract=True,
        crawl_current_sites=False,
        site_page_limit=0,
    )

    out_dir = Path(result["out_dir"])
    with (out_dir / "source_inventory.csv").open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert any(row["source_kind"] == "website_extract" for row in rows)
    assert all(str(out_dir) in row["path"] for row in rows)
    assert not any(".codex_local" in row["path"] for row in rows)


def test_final_release_extracts_local_docx_as_source(tmp_path: Path) -> None:
    google_exports = tmp_path / "gdocs"
    google_exports.mkdir()
    (google_exports / "kc_knowledge_base.txt").write_text("База знаний КЦ\nОбщее правило.\n", encoding="utf-8")
    docx_path = tmp_path / "kb.docx"
    write_minimal_docx(
        docx_path,
        [
            "Локальная база знаний КЦ",
            (
                "Нельзя обещать скидку без проверки РОПом. Если клиент спрашивает о цене, "
                "менеджер должен сверить актуальный документ и только потом отправить точный ответ."
            ),
        ],
    )

    result = build_kc_final_release(
        run_id="test_docx",
        out_dir=tmp_path / "out",
        google_export_dir=google_exports,
        old_extract_dir=tmp_path / "missing_old",
        kc_docx_path=docx_path,
        manager_patterns_path=tmp_path / "missing_patterns.jsonl",
        manager_sample_path=tmp_path / "missing_sample.jsonl",
        structured_facts_path=tmp_path / "missing_facts.jsonl",
        answer_templates_path=tmp_path / "missing_templates.csv",
        approved_answers_draft_path=tmp_path / "missing_approved.csv",
        question_items_path=tmp_path / "missing_questions.jsonl",
        fact_source_registry_path=tmp_path / "missing_registry.json",
        include_old_extract=False,
        crawl_current_sites=False,
        site_page_limit=0,
    )

    snapshot = json.loads(Path(result["snapshot_path"]).read_text(encoding="utf-8"))
    assert snapshot["summary"]["sources_by_kind"]["local_docx"] == 1
    assert any("Локальная база знаний" in chunk["text"] for chunk in snapshot["chunks"])


def test_final_release_refuses_stable_runtime_output(tmp_path: Path) -> None:
    google_exports = tmp_path / "gdocs"
    google_exports.mkdir()
    (google_exports / "kc_knowledge_base.txt").write_text("База знаний КЦ\nОбщее правило.\n", encoding="utf-8")

    with pytest.raises(ValueError, match="stable_runtime"):
        build_kc_final_release(
            run_id="test_forbidden",
            out_dir=tmp_path / "stable_runtime" / "kb",
            google_export_dir=google_exports,
            old_extract_dir=tmp_path / "missing_old",
            kc_docx_path=tmp_path / "missing.docx",
            manager_patterns_path=tmp_path / "missing_patterns.jsonl",
            manager_sample_path=tmp_path / "missing_sample.jsonl",
            structured_facts_path=tmp_path / "missing_facts.jsonl",
            answer_templates_path=tmp_path / "missing_templates.csv",
            approved_answers_draft_path=tmp_path / "missing_approved.csv",
            question_items_path=tmp_path / "missing_questions.jsonl",
            fact_source_registry_path=tmp_path / "missing_registry.json",
            include_old_extract=False,
            crawl_current_sites=False,
            site_page_limit=0,
        )


def write_minimal_docx(path: Path, paragraphs: list[str]) -> None:
    body = "".join(
        f"<w:p><w:r><w:t>{text}</w:t></w:r></w:p>"
        for text in paragraphs
    )
    document_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body>{body}</w:body>"
        "</w:document>"
    )
    with ZipFile(path, "w", ZIP_DEFLATED) as archive:
        archive.writestr("word/document.xml", document_xml)
