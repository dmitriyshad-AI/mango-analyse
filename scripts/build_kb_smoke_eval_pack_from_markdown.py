#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path(
    "/Users/dmitrijfabarisov/Claude Projects/Foton/"
    "kb_release_v2_claude_layer_2026-05-17/codex_v3_final_review/SA6_smoke_questions_50.md"
)
DEFAULT_OUT_DIR = Path("product_data/knowledge_base/kb_release_20260518_v3_2_smoke50_input")
FIELD_RE = re.compile(r"^(ID|Категория|Текст клиента|Ожидаемое поведение|Источник|Что не должно быть в draft):\s*(.*)$")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Convert Claude SA6 smoke questions markdown to Stage 6 eval inputs.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)

    result = build_smoke_eval_pack(args.input, args.out_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def build_smoke_eval_pack(input_path: Path, out_dir: Path) -> Mapping[str, Any]:
    output_root = guard_output_dir(out_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    questions = parse_smoke_markdown(input_path)
    if len(questions) != 50:
        raise ValueError(f"Expected 50 smoke questions, got {len(questions)}")

    rows = [question_to_row(item) for item in questions]
    write_jsonl(output_root / "smoke_questions_50.jsonl", rows)
    write_csv(output_root / "smoke_questions_50.csv", rows)

    by_brand = {"foton": [], "unpk": []}
    for item in rows:
        by_brand[item["brand"]].append(item)
    outputs: dict[str, str] = {}
    for brand, brand_rows in by_brand.items():
        dialog_path = output_root / f"private_dialog_threads_{brand.upper()}.jsonl"
        baseline_path = output_root / f"baseline_{brand.upper()}.csv"
        write_jsonl(dialog_path, [row_to_dialog(item) for item in brand_rows])
        write_csv(baseline_path, [row_to_baseline(item) for item in brand_rows])
        outputs[f"{brand}_dialogs"] = str(dialog_path)
        outputs[f"{brand}_baseline"] = str(baseline_path)

    return {
        "schema_version": "kb_smoke_eval_pack_v1",
        "input": str(input_path),
        "out_dir": str(output_root),
        "questions_total": len(rows),
        "by_brand": {brand: len(items) for brand, items in by_brand.items()},
        "outputs": outputs,
    }


def parse_smoke_markdown(path: Path) -> list[dict[str, str]]:
    text = path.read_text(encoding="utf-8")
    blocks = re.findall(r"```(.*?)```", text, flags=re.S)
    items: list[dict[str, str]] = []
    for block in blocks:
        fields: dict[str, str] = {}
        for raw_line in block.splitlines():
            line = raw_line.strip()
            match = FIELD_RE.match(line)
            if match:
                fields[match.group(1)] = match.group(2).strip()
        if "ID" in fields and "Текст клиента" in fields:
            items.append(fields)
    return items


def question_to_row(item: Mapping[str, str]) -> dict[str, str]:
    question_id = item["ID"]
    brand = "unpk" if question_id.startswith("unpk_") else "foton"
    expected = item.get("Ожидаемое поведение", "")
    return {
        "question_id": question_id,
        "brand": brand,
        "category": item.get("Категория", ""),
        "client_text": strip_outer_quotes(item.get("Текст клиента", "")),
        "expected_behavior": expected,
        "expected_route": infer_expected_route(expected),
        "expected_topic_id": infer_topic_id(item),
        "source_note": item.get("Источник", ""),
        "forbidden_in_draft": item.get("Что не должно быть в draft", ""),
    }


def row_to_dialog(row: Mapping[str, str]) -> dict[str, Any]:
    question_id = row["question_id"]
    current = row["client_text"]
    return {
        "dialog_id": question_id,
        "message_count": 2,
        "messages": [
            {
                "message_id": f"{question_id}:context",
                "date": "2026-05-18T10:00:00+03:00",
                "direction": "manager",
                "text": "Здравствуйте! Напишите, пожалуйста, ваш вопрос по обучению.",
            },
            {
                "message_id": f"{question_id}:client",
                "date": "2026-05-18T10:01:00+03:00",
                "direction": "client",
                "text": current,
            },
        ],
    }


def row_to_baseline(row: Mapping[str, str]) -> dict[str, str]:
    route = row["expected_route"]
    return {
        "dialog_id": row["question_id"],
        "target_message_id": f"{row['question_id']}:client",
        "topic_id": row["expected_topic_id"],
        "route": route,
        "draft_text": "Спасибо за сообщение. Передам вопрос менеджеру, он вернется с проверенным ответом."
        if route == "manager_only"
        else "Здравствуйте! Уточним актуальные условия и вернемся с ответом.",
    }


def infer_expected_route(expected: str) -> str:
    value = expected.casefold()
    if "manager_only" in value:
        return "manager_only"
    if "bot_answer_self" in value:
        return "bot_answer_self"
    return "draft_for_manager"


def infer_topic_id(item: Mapping[str, str]) -> str:
    text = " ".join(
        [
            item.get("Текст клиента", ""),
            item.get("Ожидаемое поведение", ""),
            item.get("Источник", ""),
        ]
    ).casefold().replace("ё", "е")
    if "возврат" in text or "вернуть деньги" in text:
        return "theme:009_refund"
    if "маткап" in text or "материнск" in text:
        return "theme:007_matkap_payment"
    if "налог" in text or "вычет" in text:
        return "theme:008_tax_deduction"
    if "договор" in text:
        return "theme:011_contract"
    if "справк" in text:
        return "theme:012_certificates"
    if "рассроч" in text:
        return "theme:006_installment"
    if "скид" in text or "промокод" in text:
        return "theme:005_discounts"
    if "оплат" in text or "юр.лицо" in text or "счет" in text or "счёт" in text:
        return "theme:002_payment_method"
    if "распис" in text or "дни занятия" in text or "суббот" in text or "воскрес" in text:
        return "theme:013_schedule"
    if "жалоб" in text or "преподаватель плохо" in text or "безобраз" in text:
        return "theme:019b_negative_feedback"
    if "суд" in text or "прокуратур" in text or "роспотребнадзор" in text:
        return "theme:029_legal_question"
    if "гарантир" in text:
        return "theme:019b_negative_feedback"
    if "пробн" in text:
        return "theme:023_trial_class"
    if "контакт" in text or "связаться" in text or "телефон" in text:
        return "theme:014_contacts"
    if "площадк" in text or "адрес" in text or "где проходят" in text:
        return "theme:015_location_address"
    if "преподават" in text or "кто будет вести" in text:
        return "theme:017_teachers"
    if "лагер" in text or "лвш" in text or "звш" in text or "менделеево" in text:
        return "theme:021_camps"
    if "модульн" in text or "курс" in text or "программ" in text or "направлен" in text:
        return "theme:016_program_content"
    if "сто" in text or "цен" in text:
        return "theme:001_pricing"
    return "service:S2_unclear"


def strip_outer_quotes(text: str) -> str:
    value = text.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1].strip()
    return value


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def guard_output_dir(path: Path) -> Path:
    resolved = path.expanduser().resolve(strict=False)
    if "stable_runtime" in resolved.parts:
        raise ValueError("Smoke eval output must not be inside stable_runtime")
    return resolved


if __name__ == "__main__":
    raise SystemExit(main())
