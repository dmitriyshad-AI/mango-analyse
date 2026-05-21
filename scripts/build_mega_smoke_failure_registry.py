from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


REVIEW_NOTE_TEST_IDS = {
    "matkap_foton_05",
    "address_foton_01",
    "address_foton_02",
    "camp_foton_04",
    "foton_camp_aug",
    "foton_combined_summer",
    "camp_unpk_09",
    "vk_telegram_check_foton",
}

FACT_HEAVY_CATEGORIES = {
    "pricing",
    "discount",
    "camp",
    "tax",
    "matkap",
    "schedule",
    "trial",
    "format",
    "materials",
    "results",
    "enrollment",
    "age",
    "access",
    "intensive",
    "medical",
    "reschedule",
    "contact",
    "b2b",
    "testing",
    "olympiad",
    "preschool",
    "individual",
    "transfer",
    "emergency",
    "positive",
}

GENERIC_DRAFT_RE = re.compile(
    r"(менеджер\s+(?:свяжется|подскажет|уточнит)|передам|уточн\w+\s+актуальн|"
    r"зависит\s+от|нужно\s+проверить|сориентируем\s+после)",
    re.I,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build MEGA smoke failure registry without inventing missing business facts.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = _read_json(args.run_dir / "summary.json")
    failed_rows = _read_csv(args.run_dir / "failed.csv")
    fixtures = _read_jsonl_by_id(args.fixtures) if args.fixtures else {}

    registry = [_classify_row(row, fixtures.get(row.get("test_id", ""), {})) for row in failed_rows]

    _write_csv(args.out_dir / "failure_registry.csv", registry)
    _write_baseline(args.out_dir / "baseline_summary.md", summary=summary, failed_rows=failed_rows)
    _write_registry_md(args.out_dir / "failure_registry.md", registry=registry)
    _write_questionnaire_links(args.out_dir / "questionnaire_blocking_items.md", registry=registry)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_run_dir": str(args.run_dir),
        "fixtures": str(args.fixtures) if args.fixtures else "",
        "rows_total": len(registry),
        "by_failure_class": Counter(row["failure_class"] for row in registry),
        "needs_dmitry_answer": sum(1 for row in registry if row["needs_dmitry_answer"] == "yes"),
        "blocked_waiting_business_answer": sum(
            1 for row in registry if row["proposed_fix_type"] == "blocked_waiting_business_answer"
        ),
    }
    (args.out_dir / "failure_registry_summary.json").write_text(
        json.dumps(_jsonable(manifest), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return 0


def _classify_row(row: Mapping[str, str], fixture: Mapping[str, Any]) -> dict[str, str]:
    test_id = row.get("test_id", "").strip()
    category = row.get("category", "").strip()
    subcategory = row.get("subcategory", "").strip()
    expected_route = row.get("expected_route", "").strip()
    actual_route = row.get("actual_route", "").strip()
    draft = row.get("draft_text", "").strip()
    failures = row.get("failures", "").strip()

    failure_class = "template_too_generic"
    proposed_fix = "answer_registry_template"
    needs_answer = "no"
    source_needed = ""

    if _is_obsolete_by_dmitry_20260521(test_id=test_id, category=category, failures=failures):
        failure_class = "test_obsolete_by_business_decision"
        proposed_fix = "update_mega_fixture_expected"
        needs_answer = "no"
        source_needed = "решение Дмитрия 2026-05-21: убрать устаревшие обязательные маркеры"
    elif test_id in REVIEW_NOTE_TEST_IDS or fixture.get("kb_v3_review_note"):
        failure_class = "test_requires_business_decision"
        proposed_fix = "blocked_waiting_business_answer"
        needs_answer = "yes"
        source_needed = "решение Дмитрия/РОПа по спорному тесту MEGA v3"
    elif actual_route and expected_route and actual_route != expected_route:
        if expected_route == "draft_for_manager" and actual_route == "manager_only":
            failure_class = "route_too_strict"
            proposed_fix = "route_rule_review_after_fact_confirmation"
            needs_answer = "yes" if category in FACT_HEAVY_CATEGORIES else "no"
            source_needed = "опросник: можно ли отвечать без менеджера"
        else:
            failure_class = "route_too_loose"
            proposed_fix = "semantic_gate_or_route_guard"
    elif category == "combined":
        failure_class = "multi_topic_answer_incomplete"
        proposed_fix = "answer_registry_multi_topic_template"
        needs_answer = "yes"
        source_needed = "опросник: порядок ответа на несколько тем в одном сообщении"
    elif category in {"pricing", "discount", "camp", "tax", "matkap", "schedule"}:
        failure_class = "missing_fact" if _needs_precise_fact(failures) else "template_too_generic"
        proposed_fix = "blocked_waiting_business_answer" if failure_class == "missing_fact" else "answer_registry_template"
        needs_answer = "yes" if failure_class == "missing_fact" else "no"
        source_needed = "опросник и подтверждённый источник факта" if needs_answer == "yes" else "answer_registry"
    elif category == "adversarial":
        failure_class = "template_too_generic" if GENERIC_DRAFT_RE.search(draft) else "expected_marker_too_strict"
        proposed_fix = "semantic_gate_or_safe_template"
    elif category in FACT_HEAVY_CATEGORIES:
        failure_class = "template_too_generic" if GENERIC_DRAFT_RE.search(draft) else "missing_fact"
        proposed_fix = "answer_registry_template" if failure_class == "template_too_generic" else "blocked_waiting_business_answer"
        needs_answer = "yes" if failure_class == "missing_fact" else "no"
        source_needed = "опросник и подтверждённый источник факта" if needs_answer == "yes" else "answer_registry"
    if failures == "missing_expected=менеджер" and expected_route == actual_route:
        failure_class = "expected_marker_too_strict"
        proposed_fix = "test_expected_review"
        needs_answer = "no"
        source_needed = "проверить wording теста"

    missing_expected = failures.replace("missing_expected=", "").replace("; ", "|")
    return {
        "test_id": test_id,
        "brand": row.get("brand", ""),
        "priority": row.get("priority", ""),
        "category": category,
        "subcategory": subcategory,
        "client_message": row.get("client_message", ""),
        "expected_route": expected_route,
        "actual_route": actual_route,
        "missing_expected": missing_expected,
        "draft_text": draft,
        "failure_class": failure_class,
        "needs_dmitry_answer": needs_answer,
        "proposed_fix_type": proposed_fix,
        "source_needed": source_needed,
    }


def _needs_precise_fact(failures: str) -> bool:
    return bool(re.search(r"\d|%|руб|₽|мая|июня|июля|августа|дней|недель|адрес|VK|Telegram|МТС", failures, re.I))


def _is_obsolete_by_dmitry_20260521(*, test_id: str, category: str, failures: str) -> bool:
    lowered = failures.casefold()
    if category in {"high_risk", "adversarial"} and (
        "автоматический ответ" in lowered or "зафиксировано" in lowered
    ):
        return True
    if test_id == "address_unpk_01" and "лобня" in lowered:
        return True
    return False


def _write_baseline(path: Path, *, summary: Mapping[str, Any], failed_rows: list[dict[str, str]]) -> None:
    by_category = Counter(row.get("category", "") for row in failed_rows)
    by_brand = Counter(row.get("brand", "") for row in failed_rows)
    lines = [
        "# Baseline MEGA smoke v3",
        "",
        f"- source: `mega_smoke_v3_codex_full_all_20260520_112720`",
        f"- total: `{summary.get('total_completed')}/{summary.get('total_expected_rows')}`",
        f"- passed: `{summary.get('passed')}`",
        f"- failed: `{summary.get('failed')}`",
        f"- pass_rate: `{summary.get('pass_rate')}`",
        f"- P0: `{summary.get('by_priority', {}).get('P0', {}).get('passed')}/157 passed`",
        f"- P1: `{summary.get('by_priority', {}).get('P1', {}).get('passed')}/122 passed`",
        f"- P2: `{summary.get('by_priority', {}).get('P2', {}).get('passed')}/2 passed`",
        "",
        "## Failed by brand",
        "",
    ]
    lines.extend(f"- `{brand}`: {count}" for brand, count in by_brand.most_common())
    lines.extend(["", "## Failed by category", ""])
    lines.extend(f"- `{category}`: {count}" for category, count in by_category.most_common())
    lines.extend(
        [
            "",
            "## Rule",
            "",
            "Факты не выдумываются. Если нет подтверждения в опроснике или источнике, строка получает `blocked_waiting_business_answer`.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_registry_md(path: Path, *, registry: list[dict[str, str]]) -> None:
    by_class = Counter(row["failure_class"] for row in registry)
    lines = ["# Failure registry", "", "## Summary", ""]
    lines.extend(f"- `{key}`: {value}" for key, value in by_class.most_common())
    lines.extend(["", "## Rows", ""])
    for row in registry:
        lines.append(
            f"- `{row['test_id']}` [{row['brand']}/{row['category']}/{row['subcategory']}]: "
            f"`{row['failure_class']}`, fix=`{row['proposed_fix_type']}`, Дмитрий=`{row['needs_dmitry_answer']}`"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_questionnaire_links(path: Path, *, registry: list[dict[str, str]]) -> None:
    blocked = [row for row in registry if row["needs_dmitry_answer"] == "yes"]
    lines = [
        "# Что ждёт ответов из опросника",
        "",
        "Эти строки нельзя закрывать догадками. До ответа команды использовать safe-handoff.",
        "",
    ]
    for row in blocked:
        lines.append(
            f"- `{row['test_id']}` [{row['brand']}/{row['category']}/{row['subcategory']}]: "
            f"{row['source_needed']} | вопрос клиента: {row['client_message']}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _read_jsonl_by_id(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            item = json.loads(line)
            records[str(item.get("id") or item.get("test_id") or "")] = item
    return records


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "test_id",
        "brand",
        "priority",
        "category",
        "subcategory",
        "client_message",
        "expected_route",
        "actual_route",
        "missing_expected",
        "draft_text",
        "failure_class",
        "needs_dmitry_answer",
        "proposed_fix_type",
        "source_needed",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Counter):
        return dict(value)
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    return value


if __name__ == "__main__":
    raise SystemExit(main())
