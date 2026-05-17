from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.build_full_kc_knowledge_base import build_full_kc_knowledge_base


def test_full_kb_marks_precise_facts_unverified_and_scripts_draft_only(tmp_path: Path) -> None:
    input_dir = tmp_path / "exports"
    out_dir = tmp_path / "kb"
    input_dir.mkdir()
    (input_dir / "foton_prices_2026_2027.txt").write_text(
        "\n".join(
            [
                "Стоимость обучения и порядок оплаты на 2026/2027 уч.г.",
                "5-11 классы",
                "Стоимость 49 000 руб.",
                "Скидка 20% на второй предмет",
                "Оплата до 15 мая.",
            ]
        ),
        encoding="utf-8",
    )
    (input_dir / "call_scripts.txt").write_text(
        "\n".join(
            [
                "Скрипт звонка",
                "[Имя Отчество], добрый день!",
                "Подскажите, вам удобно сейчас разговаривать?",
                "Обязательно дослушиваем родителя и задаем уточняющие вопросы.",
            ]
        ),
        encoding="utf-8",
    )

    result = build_full_kc_knowledge_base(input_dir=input_dir, out_dir=out_dir, manager_patterns_path=tmp_path / "missing.jsonl")

    assert result["sources_total"] == 2
    assert result["processed_sources"] == 2
    assert result["fact_candidates"] >= 3
    assert result["conversation_scripts"] >= 1
    facts = [json.loads(line) for line in (out_dir / "fact_candidates.jsonl").read_text(encoding="utf-8").splitlines()]
    assert all(fact["usable_for_precise_answer"] is False for fact in facts)
    assert all(fact["requires_manager_confirmation"] is True for fact in facts)
    assert all(fact["bot_permission"] == "manager_only" for fact in facts)
    scripts = [json.loads(line) for line in (out_dir / "conversation_scripts.jsonl").read_text(encoding="utf-8").splitlines()]
    assert all(script["bot_permission"] == "draft_for_manager" for script in scripts)


def test_full_kb_records_html_export_as_problem_source(tmp_path: Path) -> None:
    input_dir = tmp_path / "exports"
    out_dir = tmp_path / "kb"
    input_dir.mkdir()
    (input_dir / "bad_doc.txt").write_text("<!DOCTYPE html><html><body>Нет доступа</body></html>", encoding="utf-8")

    result = build_full_kc_knowledge_base(input_dir=input_dir, out_dir=out_dir, manager_patterns_path=tmp_path / "missing.jsonl")

    assert result["sources_total"] == 1
    assert result["processed_sources"] == 0
    with (out_dir / "source_inventory.csv").open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert rows[0]["processing_status"] == "export_failed"
    assert "html" in rows[0]["status_reason"]


def test_full_kb_refuses_stable_runtime_output(tmp_path: Path) -> None:
    input_dir = tmp_path / "exports"
    input_dir.mkdir()
    (input_dir / "call_scripts.txt").write_text("Скрипт звонка\nДобрый день!", encoding="utf-8")

    with pytest.raises(ValueError, match="stable_runtime"):
        build_full_kc_knowledge_base(
            input_dir=input_dir,
            out_dir=tmp_path / "stable_runtime" / "kb",
            manager_patterns_path=tmp_path / "missing.jsonl",
        )
