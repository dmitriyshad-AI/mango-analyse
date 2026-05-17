from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

from mango_mvp.knowledge_base.manager_answer_playbook import (
    CLASSIFICATION_GOOD,
    CLASSIFICATION_NO_ANSWER,
    CLASSIFICATION_OUTDATED,
    CLASSIFICATION_UNSAFE,
    build_manager_answer_playbook,
    sanitize_manager_text,
)


def test_build_playbook_writes_safe_outputs(tmp_path: Path) -> None:
    catalog_root = _write_catalog(tmp_path)
    out_dir = tmp_path / "kb"

    payload = build_manager_answer_playbook(
        catalog_root,
        out_dir,
        sample_size=20,
        min_sample_size=10,
        pattern_limit=20,
        write_xlsx=False,
    )

    assert payload["mode"] == "read_only"
    assert payload["safety"]["manager_answers_are_facts"] is False
    assert payload["summary"]["answers_usable_as_fact"] == 0
    assert payload["summary"]["patterns_usable_as_fact"] == 0

    sample_rows = _read_csv(out_dir / "manager_answer_sample_300_500.csv")
    assert 10 <= len(sample_rows) <= 20
    assert {"call", "telegram", "email"} <= {row["channel"] for row in sample_rows}
    assert {"цена", "расписание", "доступ/ссылки"} <= {row["topic"] for row in sample_rows}
    assert {
        CLASSIFICATION_GOOD,
        CLASSIFICATION_OUTDATED,
        CLASSIFICATION_UNSAFE,
        CLASSIFICATION_NO_ANSWER,
    } <= {row["answer_classification"] for row in sample_rows}
    assert all(row["usable_as_fact"] == "false" for row in sample_rows)

    sample_text = (out_dir / "manager_answer_sample_300_500.csv").read_text(encoding="utf-8-sig")
    patterns_text = (out_dir / "manager_answer_patterns.jsonl").read_text(encoding="utf-8")
    assert "ivan@example.com" not in sample_text
    assert "89991234567" not in sample_text
    assert "50000" not in sample_text
    assert "31.05" not in sample_text
    assert "50000" not in patterns_text

    patterns = [json.loads(line) for line in patterns_text.splitlines() if line.strip()]
    assert patterns
    assert all(pattern["usable_as_fact"] is False for pattern in patterns)
    assert all("Historical manager answers are style examples only" in pattern["fact_safety_note"] for pattern in patterns)

    playbook_md = (out_dir / "manager_answer_playbook.md").read_text(encoding="utf-8")
    assert "это не база фактов" in playbook_md
    assert "Не копировать исторические ответы клиенту дословно" in playbook_md
    assert (out_dir / "unsafe_or_outdated_manager_answers.csv").exists()
    assert (out_dir / "manager_answer_sample_300_500.jsonl").exists()


def test_sanitize_manager_text_removes_personal_and_dynamic_values() -> None:
    text = "Пишите ivan@example.com или 89991234567, сумма 50000 рублей, скидка 10% до 31.05 в 18:00."

    sanitized = sanitize_manager_text(text)

    assert "ivan@example.com" not in sanitized
    assert "89991234567" not in sanitized
    assert "50000" not in sanitized
    assert "10%" not in sanitized
    assert "31.05" not in sanitized
    assert "[email]" in sanitized
    assert "[телефон]" in sanitized
    assert "[сумма]" in sanitized
    assert "[процент]" in sanitized
    assert "[дата]" in sanitized
    assert "[время]" in sanitized


def test_build_manager_answer_playbook_script(tmp_path: Path) -> None:
    catalog_root = _write_catalog(tmp_path)
    out_dir = tmp_path / "script_out"
    env = {**os.environ, "PYTHONPATH": "src"}

    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_manager_answer_playbook.py",
            "--catalog-root",
            str(catalog_root),
            "--out-dir",
            str(out_dir),
            "--sample-size",
            "20",
            "--min-sample-size",
            "10",
            "--skip-xlsx",
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert payload["summary"]["sample_rows"] >= 10
    assert payload["safety"]["client_send"] is False
    assert Path(payload["outputs"]["sample_csv"]).exists()
    assert Path(payload["outputs"]["patterns_jsonl"]).exists()
    assert Path(payload["outputs"]["playbook_md"]).exists()


def _write_catalog(tmp_path: Path) -> Path:
    root = tmp_path / "question_catalog"
    root.mkdir(parents=True)
    classes = [
        _class_row("class:price", "стоимость", "базовая стоимость", "prices.current", "allowed_after_fact_check", "template_ready_needs_current_fact"),
        _class_row("class:schedule", "расписание", "день и время занятия", "schedule.current", "allowed_after_fact_check", "template_ready_needs_current_fact"),
        _class_row("class:payment", "оплата / возврат / чек", "способ оплаты", "", "draft_only_needs_review", "draft_answer_exists_needs_review"),
        _class_row("class:docs", "письма / справки / подтверждающие документы", "справка", "documents.current", "manager_only", "manager_only"),
        _class_row("class:matcap", "оплата / возврат / чек", "оплата материнским капиталом", "documents.current", "manager_only", "manager_only"),
        _class_row("class:tax", "письма / справки / подтверждающие документы", "налоговый вычет", "documents.current", "manager_only", "needs_rop_answer"),
        _class_row("class:refund", "оплата / возврат / чек", "возврат оплаты", "documents.current", "manager_only", "manager_only"),
        _class_row("class:trial", "запись на обучение", "пробное занятие", "", "draft_only_needs_review", "draft_answer_exists_needs_review"),
        _class_row("class:program", "программа курса", "содержание курса", "", "draft_only_needs_review", "draft_answer_exists_needs_review"),
        _class_row("class:access", "доступ / технический вопрос", "ссылка не пришла", "", "draft_only_needs_review", "draft_answer_exists_needs_review"),
        _class_row("class:complaint", "качество обучения / обратная связь", "жалоба", "", "draft_only_needs_review", "draft_answer_exists_needs_review"),
        _class_row("class:other", "общий вопрос", "без контекста", "", "not_allowed", "not_enough_context"),
    ]
    _write_csv(root / "customer_question_classes.csv", classes)
    _write_csv(
        root / "question_answer_quality_review_2026-05-14_final.csv",
        [
            {
                "ID класса": "class:docs",
                "Исторический ответ менеджера (не утверждено)": "",
                "Блокер качества": "",
                "Нужные актуальные факты": "documents.current",
                "Группы риска": "документы",
                "Риск ошибки": "высокий",
                "Крупный класс": "документы",
                "Узкий класс": "справка",
            }
        ],
    )
    _write_csv(
        root / "approved_question_answers_draft.csv",
        [
            {
                "question_class_id": "class:payment",
                "required_fact_keys": "",
                "runtime_bot_permission": "draft_only_needs_review",
                "bot_permission": "draft_only_needs_review",
            }
        ],
    )
    _write_jsonl(
        root / "customer_question_items.jsonl",
        [
            _item("q1", "class:price", "call", "Сколько стоит курс?", "Стоимость 50000 рублей до 31.05.", "template_ready_needs_current_fact", ["price"], True),
            _item("q2", "class:schedule", "telegram", "Когда занятие?", "Группа идет по средам в 18:00.", "template_ready_needs_current_fact", ["schedule"], True),
            _item("q3", "class:payment", "email", "Как оплатить?", "Понял вопрос по оплате, уточните номер договора, менеджер подскажет следующий шаг.", "draft_answer_exists_needs_review", [], False),
            _item("q4", "class:docs", "call", "Нужна справка.", "Мы гарантируем, что оформим документ.", "manager_only", ["documents"], True),
            _item("q5", "class:matcap", "telegram", "Можно маткапиталом?", "Можно оплатить маткапиталом, оформим.", "manager_only", ["documents"], True),
            _item("q6", "class:tax", "email", "Как получить вычет?", "", "needs_rop_answer", ["documents"], True),
            _item("q7", "class:refund", "call", "Вернете деньги?", "Вернем 10000 рублей.", "manager_only", ["documents"], True),
            _item("q8", "class:trial", "telegram", "Можно пробное?", "Уточните класс, предмет и формат, менеджер подберет вариант.", "draft_answer_exists_needs_review", [], False),
            _item("q9", "class:program", "email", "Что в программе?", "Менеджер сверит программу под ваш класс и предмет.", "draft_answer_exists_needs_review", [], False),
            _item("q10", "class:access", "call", "Не пришла ссылка.", "Напишите на ivan@example.com или 89991234567, проверим.", "draft_answer_exists_needs_review", [], False, ["email_redacted"]),
            _item("q11", "class:complaint", "telegram", "Я недовольна уроком.", "Понимаю, ситуация неприятная. Передам менеджеру, он проверит детали.", "draft_answer_exists_needs_review", [], False),
            _item("q12", "class:other", "email", "Спасибо", "угу", "not_enough_context", [], False),
        ],
    )
    return root


def _class_row(
    class_id: str,
    parent: str,
    subclass: str,
    fact_keys: str,
    bot_permission: str,
    answer_status: str,
) -> dict[str, str]:
    return {
        "question_class_id": class_id,
        "parent_question_class": parent,
        "question_subclass": subclass,
        "canonical_question": f"{parent} / {subclass}",
        "required_fact_keys": fact_keys,
        "bot_permission": bot_permission,
        "answer_status": answer_status,
    }


def _item(
    item_id: str,
    class_id: str,
    channel: str,
    question: str,
    answer: str,
    status: str,
    dynamic_fact_types: list[str],
    requires_dynamic_facts: bool,
    safety_flags: list[str] | None = None,
) -> dict:
    return {
        "question_item_id": item_id,
        "question_class_id": class_id,
        "source_channel": channel,
        "source_ref": f"{channel}:{item_id}",
        "occurred_at": "2026-05-14T10:00:00+00:00",
        "customer_text_redacted": question,
        "manager_text_redacted": answer,
        "answer_evidence_status": status,
        "answer_source": "test",
        "dynamic_fact_types": dynamic_fact_types,
        "requires_dynamic_facts": requires_dynamic_facts,
        "safety_flags": safety_flags or [],
        "intent": "",
        "product": "регулярный курс",
        "format": "",
        "metadata": {
            "answer_status": status,
            "bot_permission": "manager_only" if status == "manager_only" else "draft_only_needs_review",
            "required_fact_keys": "|".join(f"{fact}.current" for fact in dynamic_fact_types),
        },
    }


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))
