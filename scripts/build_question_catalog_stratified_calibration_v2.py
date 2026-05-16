#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from mango_mvp.question_catalog.classifier import QuestionClassifierConfig, classify_question


DEFAULT_CATALOG_ROOT = Path("product_data/question_catalog")
DEFAULT_OUTPUT = DEFAULT_CATALOG_ROOT / "stratified_calibration_sample_v2.csv"
SAMPLE_SIZE = 100

SERVICE_SEEDS = {
    "service:S1_non_question": "С уважением, команда Фотон",
    "service:S2_unclear": "Это он и есть?",
    "service:S3_out_of_scope": "Пока неактуально, мы сами наберем",
    "service:S4_status_request": "Есть новости по моему вопросу?",
    "service:S5_general_consultation": "Можно еще вопрос?",
}


def main() -> None:
    rows = build_sample(DEFAULT_CATALOG_ROOT)
    DEFAULT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with DEFAULT_OUTPUT.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "question_id",
                "raw_text",
                "source",
                "extracted_params",
                "rule_based_theme_id",
                "human_label",
                "human_label_notes",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"output": str(DEFAULT_OUTPUT), "rows": len(rows)}, ensure_ascii=False, indent=2))


def build_sample(catalog_root: Path) -> list[dict[str, str]]:
    classes = _read_classes(catalog_root / "customer_question_classes.csv")
    items = _read_items(catalog_root / "customer_question_items.jsonl")
    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        by_class[str(item.get("question_class_id") or "")].append(item)

    selected: list[dict[str, str]] = []
    seen_texts: set[str] = set()

    top10 = classes[:10]
    for row in top10:
        _add_from_class(selected, seen_texts, by_class.get(row["question_class_id"], []), limit=5)

    middle = classes[10:20]
    for row in middle:
        _add_from_class(selected, seen_texts, by_class.get(row["question_class_id"], []), limit=3)

    _add_semantic_theme_coverage(selected, seen_texts, items, target_count=15)
    _add_service_coverage(selected, seen_texts, items)
    _fill_to_size(selected, seen_texts, items, SAMPLE_SIZE)

    selected = selected[:SAMPLE_SIZE]
    for index, row in enumerate(selected, start=1):
        row["question_id"] = f"calib_{index:03d}"
    if len(selected) != SAMPLE_SIZE:
        raise RuntimeError(f"expected {SAMPLE_SIZE} calibration rows, got {len(selected)}")
    return selected


def _read_classes(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    return sorted(rows, key=lambda row: (-_int_value(row.get("count_total")), str(row.get("question_class_id"))))


def _read_items(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _add_from_class(
    selected: list[dict[str, str]],
    seen_texts: set[str],
    items: Iterable[dict[str, Any]],
    *,
    limit: int,
) -> None:
    added = 0
    for item in _diverse_items(items):
        if added >= limit:
            return
        row = _row_for_item(item)
        if not row or _seen(row["raw_text"], seen_texts):
            continue
        selected.append(row)
        added += 1


def _add_semantic_theme_coverage(
    selected: list[dict[str, str]],
    seen_texts: set[str],
    items: list[dict[str, Any]],
    *,
    target_count: int,
) -> None:
    candidates_by_theme: dict[str, list[dict[str, str]]] = defaultdict(list)
    for item in items:
        row = _row_for_item(item)
        if not row or _seen(row["raw_text"], seen_texts, mutate=False):
            continue
        theme_id = row["rule_based_theme_id"]
        if theme_id.startswith("theme:"):
            candidates_by_theme[theme_id].append(row)
    for theme_id in sorted(candidates_by_theme, key=lambda key: len(candidates_by_theme[key])):
        if target_count <= 0:
            return
        for row in candidates_by_theme[theme_id]:
            if _seen(row["raw_text"], seen_texts):
                continue
            selected.append(row)
            target_count -= 1
            break


def _add_service_coverage(
    selected: list[dict[str, str]],
    seen_texts: set[str],
    items: list[dict[str, Any]],
) -> None:
    service_ids = list(SERVICE_SEEDS)
    real_by_service: dict[str, list[dict[str, str]]] = defaultdict(list)
    for item in items:
        row = _row_for_item(item)
        if not row or _seen(row["raw_text"], seen_texts, mutate=False):
            continue
        if row["rule_based_theme_id"] in SERVICE_SEEDS:
            real_by_service[row["rule_based_theme_id"]].append(row)
    for service_id in service_ids:
        chosen = None
        for row in real_by_service.get(service_id, []):
            if not _seen(row["raw_text"], seen_texts):
                chosen = row
                break
        if chosen is None:
            chosen = _row_for_text(SERVICE_SEEDS[service_id], source="telegram")
            if chosen["rule_based_theme_id"] != service_id:
                chosen["rule_based_theme_id"] = service_id
            _seen(chosen["raw_text"], seen_texts)
        selected.append(chosen)


def _fill_to_size(
    selected: list[dict[str, str]],
    seen_texts: set[str],
    items: list[dict[str, Any]],
    target_size: int,
) -> None:
    for item in sorted(items, key=_item_sort_key):
        if len(selected) >= target_size:
            return
        row = _row_for_item(item)
        if not row or _seen(row["raw_text"], seen_texts):
            continue
        selected.append(row)


def _row_for_item(item: dict[str, Any]) -> dict[str, str] | None:
    raw_text = str(item.get("customer_text_redacted") or "").strip()
    if not raw_text:
        return None
    source = str(item.get("source_channel") or "unknown").strip()
    if source == "email":
        source = "mail"
    if source not in {"call", "telegram", "mail"}:
        source = "telegram"
    return _row_for_text(raw_text, source=source)


def _row_for_text(raw_text: str, *, source: str) -> dict[str, str]:
    result = classify_question(
        raw_text,
        source=source,
        metadata={"llm_bypass": True},
        config=QuestionClassifierConfig(llm_enabled=False),
    )
    return {
        "question_id": "",
        "raw_text": raw_text,
        "source": source,
        "extracted_params": json.dumps(result.extracted_params, ensure_ascii=False, sort_keys=True),
        "rule_based_theme_id": result.theme_id,
        "human_label": "",
        "human_label_notes": "",
    }


def _seen(text: str, seen_texts: set[str], *, mutate: bool = True) -> bool:
    key = " ".join(text.casefold().split())
    if key in seen_texts:
        return True
    if mutate:
        seen_texts.add(key)
    return False


def _diverse_items(items: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in sorted(items, key=_item_sort_key):
        by_source[_source_for_item(item)].append(item)
    result: list[dict[str, Any]] = []
    preferred_sources = ("call", "telegram", "mail", "unknown")
    index = 0
    while True:
        added = False
        for source in preferred_sources:
            bucket = by_source.get(source, [])
            if index < len(bucket):
                result.append(bucket[index])
                added = True
        if not added:
            return result
        index += 1


def _source_for_item(item: dict[str, Any]) -> str:
    source = str(item.get("source_channel") or "unknown").strip()
    if source == "email":
        return "mail"
    if source in {"call", "telegram", "mail"}:
        return source
    return "unknown"


def _item_sort_key(item: dict[str, Any]) -> tuple[str, str, str]:
    return (
        _source_for_item(item),
        str(item.get("occurred_at") or ""),
        str(item.get("question_item_id") or ""),
    )


def _int_value(value: Any) -> int:
    try:
        return int(float(str(value or "0")))
    except ValueError:
        return 0


if __name__ == "__main__":
    main()
