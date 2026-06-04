#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCHEDULE_PACK = PROJECT_ROOT / "audits" / "_inbox" / "schedule_vs_tallanto_recheck_20260602_123118"
DEFAULT_SOURCE_DIR = PROJECT_ROOT / "product_data" / "knowledge_base" / "kb_release_20260520_v6_3_team_answers_sources"
DEFAULT_RELEASE_ID = "kb_release_20260603_v6_5_summer_format_cleanup"
DEFAULT_FRESHNESS_DATE = "2026-06-02"
GENERATED_BEGIN = "# BEGIN GENERATED schedule_2026_27 by scripts/derive_kb_schedule_2026_27_sources.py"
GENERATED_END = "# END GENERATED schedule_2026_27"


BRAND_TO_SOURCE = {
    "Фотон": "foton",
    "УНПК МФТИ": "unpk",
}
BRAND_TO_FILE = {
    "foton": "facts_for_bot_FOTON.yaml",
    "unpk": "facts_for_bot_UNPK.yaml",
}
DAY_SLUGS = {
    "Пн": "mon",
    "Вт": "tue",
    "Ср": "wed",
    "Чт": "thu",
    "Пт": "fri",
    "Сб": "sat",
    "Вс": "sun",
    "Пн+Ср": "mon_wed",
    "Вт+Чт": "tue_thu",
}
TRACK_PHRASES = {
    "ЕГЭ": "ЕГЭ",
    "ОГЭ": "ОГЭ",
    "продвинутая": "продвинутая группа",
    "базовая": "базовая группа",
    "олимпиадная": "олимпиадная группа",
    "обычная": "обычная группа",
}


class DerivationError(ValueError):
    pass


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Derive verified 2026/27 schedule facts into KB YAML sources.")
    parser.add_argument("--schedule-pack", type=Path, default=DEFAULT_SCHEDULE_PACK)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--audit-pack", type=Path, required=True)
    parser.add_argument("--release-id", default=DEFAULT_RELEASE_ID)
    parser.add_argument("--freshness-date", default=DEFAULT_FRESHNESS_DATE)
    args = parser.parse_args(argv)

    schedule_pack = args.schedule_pack.expanduser().resolve(strict=False)
    source_dir = args.source_dir.expanduser().resolve(strict=False)
    audit_pack = args.audit_pack.expanduser().resolve(strict=False)
    audit_pack.mkdir(parents=True, exist_ok=True)

    comparison_payload = load_json(schedule_pack / "schedule_vs_tallanto_comparison.json")
    comparisons = comparison_payload.get("comparisons")
    if not isinstance(comparisons, list):
        raise DerivationError("schedule_vs_tallanto_comparison.json must contain comparisons list.")
    summary = as_mapping(comparison_payload.get("summary"))
    validate_clean_reconciliation(summary, comparisons)

    tallanto_rows = load_json(schedule_pack / "tallanto_schedule_normalized.json")
    if not isinstance(tallanto_rows, list):
        raise DerivationError("tallanto_schedule_normalized.json must contain a list.")
    tallanto_by_key = {str(row.get("match_key") or ""): row for row in tallanto_rows if isinstance(row, Mapping)}

    facts_by_brand: dict[str, dict[str, Any]] = {"foton": {}, "unpk": {}}
    ingest_rows: list[dict[str, str]] = []
    examples: dict[str, list[str]] = {"foton": [], "unpk": []}
    empty_filial_group_ids: list[str] = []

    for row in comparisons:
        if not isinstance(row, Mapping):
            continue
        match_key = str(row.get("match_key") or "")
        status = str(row.get("status") or "")
        tallanto_row = tallanto_by_key.get(match_key) or {}
        tallanto_matches = row.get("tallanto_matches") if isinstance(row.get("tallanto_matches"), list) else []
        match = tallanto_matches[0] if tallanto_matches and isinstance(tallanto_matches[0], Mapping) else {}
        group_id = str(match.get("id") or tallanto_row.get("id") or "")
        decision = "include" if status == "совпало" else "exclude"
        note_parts: list[str] = []
        raw_record = tallanto_row.get("raw_record") if isinstance(tallanto_row.get("raw_record"), Mapping) else {}
        if group_id and not raw_record.get("filial"):
            note_parts.append("empty_filial_fallback_by_name")
            empty_filial_group_ids.append(group_id)
        ingest_rows.append(
            {
                "group_id": group_id,
                "status": status,
                "decision": decision,
                "brand": str(row.get("brand") or ""),
                "format": str(row.get("format") or ""),
                "place": str(row.get("place") or ""),
                "day": str(row.get("day") or ""),
                "time": str(row.get("time") or ""),
                "subject": str(row.get("subject") or ""),
                "grade": str(row.get("grade") or ""),
                "track": str(row.get("track") or ""),
                "date_start": str(match.get("date_start") or tallanto_row.get("date_start") or ""),
                "date_finish": str(match.get("date_finish") or tallanto_row.get("date_finish") or ""),
                "sheet_cell": str(row.get("sheet_cell") or ""),
                "match_key": match_key,
                "note": ";".join(note_parts),
            }
        )
        if decision != "include":
            continue
        fact = build_schedule_fact(row, tallanto_row, match)
        brand_key = BRAND_TO_SOURCE.get(str(row.get("brand") or ""))
        if brand_key not in facts_by_brand:
            raise DerivationError(f"Unsupported brand in comparison row: {row.get('brand')!r}")
        facts_by_brand[brand_key][fact["key"]] = {"client_safe_text": fact["client_safe_text"]}
        if len(examples[brand_key]) < 4:
            examples[brand_key].append(fact["client_safe_text"])

    write_ingest_decision(audit_pack / "ingest_decision.csv", ingest_rows)
    write_source_sections(
        source_dir=source_dir,
        facts_by_brand=facts_by_brand,
        release_id=args.release_id,
        freshness_date=args.freshness_date,
        audit_pack=audit_pack,
        schedule_pack=schedule_pack,
    )
    write_derivation_reports(
        audit_pack=audit_pack,
        schedule_pack=schedule_pack,
        source_dir=source_dir,
        release_id=args.release_id,
        freshness_date=args.freshness_date,
        summary=summary,
        ingest_rows=ingest_rows,
        facts_by_brand=facts_by_brand,
        examples=examples,
        empty_filial_group_ids=empty_filial_group_ids,
    )
    print(
        json.dumps(
            {
                "release_id": args.release_id,
                "schedule_pack": str(schedule_pack),
                "source_dir": str(source_dir),
                "audit_pack": str(audit_pack),
                "facts_by_brand": {brand: len(items) for brand, items in facts_by_brand.items()},
                "ingest_rows": len(ingest_rows),
                "empty_filial_fallback_groups": empty_filial_group_ids,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def validate_clean_reconciliation(summary: Mapping[str, Any], comparisons: Sequence[Mapping[str, Any]]) -> None:
    expected = {
        "mismatched_field": 0,
        "missing_in_tallanto": 0,
        "duplicate_tallanto_matches": 0,
        "extra_in_tallanto": 0,
    }
    for key, expected_value in expected.items():
        if int(summary.get(key) or 0) != expected_value:
            raise DerivationError(f"Input gate is not clean: {key}={summary.get(key)!r}.")
    statuses = Counter(str(row.get("status") or "") for row in comparisons if isinstance(row, Mapping))
    if set(statuses) != {"совпало"}:
        raise DerivationError(f"Input gate is not clean: status counts are {dict(statuses)!r}.")
    if int(summary.get("matched") or 0) != len(comparisons):
        raise DerivationError(f"Input gate mismatch: matched={summary.get('matched')!r}, rows={len(comparisons)}.")


def build_schedule_fact(row: Mapping[str, Any], tallanto_row: Mapping[str, Any], match: Mapping[str, Any]) -> dict[str, str]:
    date_start = str(match.get("date_start") or tallanto_row.get("date_start") or "").strip()
    date_finish = str(match.get("date_finish") or tallanto_row.get("date_finish") or "").strip()
    if not date_start or not date_finish:
        raise DerivationError(f"Missing start/finish dates for {row.get('match_key')!r}.")
    valid_slug = date_finish.replace("-", "_")
    key = "_".join(
        part
        for part in (
            "group",
            "start_date",
            safe_slug(str(row.get("sheet_cell") or "")),
            safe_slug(str(tallanto_row.get("filial_key") or "")),
            DAY_SLUGS.get(str(tallanto_row.get("day") or ""), safe_slug(str(tallanto_row.get("day") or ""))),
            time_slug(str(row.get("time") or "")),
            safe_slug(str(tallanto_row.get("subject") or "")),
            safe_slug(str(row.get("grade") or "")),
            safe_slug(str(tallanto_row.get("track") or "")),
            f"before_{valid_slug}",
        )
        if part
    )
    client_safe_text = render_client_safe_text(row, date_start)
    return {"key": key, "client_safe_text": client_safe_text}


def render_client_safe_text(row: Mapping[str, Any], date_start: str) -> str:
    subject = capitalize_first(str(row.get("subject") or "").strip())
    grade = str(row.get("grade") or "").strip()
    grade_phrase = f"{grade} класс" if grade else "класс уточняется"
    track = TRACK_PHRASES.get(str(row.get("track") or "").strip(), str(row.get("track") or "").strip())
    track_phrase = f", {track}" if track else ""
    fmt = str(row.get("format") or "").strip()
    place = str(row.get("place") or "").strip()
    day = str(row.get("day") or "").strip()
    time = str(row.get("time") or "").strip()
    start = format_date_ru(date_start)
    return (
        f"{subject}, {grade_phrase}{track_phrase}, {fmt}, {place}: "
        f"{day} {time}, старт {start}. "
        "Точное расписание конкретной группы уточняется."
    )


def write_source_sections(
    *,
    source_dir: Path,
    facts_by_brand: Mapping[str, Mapping[str, Any]],
    release_id: str,
    freshness_date: str,
    audit_pack: Path,
    schedule_pack: Path,
) -> None:
    facts_dir = source_dir / "facts"
    for brand, groups in facts_by_brand.items():
        filename = BRAND_TO_FILE[brand]
        path = facts_dir / filename
        if not path.exists():
            raise FileNotFoundError(path)
        block_payload = {
            "schedule_2026_27": {
                "status": "verified",
                "freshness_status": "fresh_verified",
                "client_facing": True,
                "source": (
                    f"Google Sheet 2026-2027 + Tallanto most_courses read-only reconciliation "
                    f"{freshness_date}; source_pack={schedule_pack}; audit_pack={audit_pack}"
                ),
                "groups": dict(groups),
            }
        }
        block = yaml.safe_dump(block_payload, allow_unicode=True, sort_keys=False).rstrip()
        replace_generated_block(path, block)
    update_manifest(source_dir / "release_manifest.yaml", release_id=release_id, freshness_date=freshness_date)


def replace_generated_block(path: Path, yaml_block: str) -> None:
    original = path.read_text(encoding="utf-8")
    generated = f"{GENERATED_BEGIN}\n{yaml_block}\n{GENERATED_END}\n"
    pattern = re.compile(
        rf"\n?{re.escape(GENERATED_BEGIN)}\n.*?\n{re.escape(GENERATED_END)}\n?",
        flags=re.DOTALL,
    )
    if pattern.search(original):
        updated = pattern.sub("\n" + generated, original).rstrip() + "\n"
    else:
        if re.search(r"^schedule_2026_27:", original, flags=re.MULTILINE):
            raise DerivationError(f"{path} already contains unmarked schedule_2026_27 section.")
        updated = original.rstrip() + "\n\n" + generated
    path.write_text(updated, encoding="utf-8")


def update_manifest(path: Path, *, release_id: str, freshness_date: str) -> None:
    manifest = load_yaml(path)
    manifest["release_id"] = release_id
    manifest["freshness_check_date"] = freshness_date
    required_paths = list(manifest.get("required_yaml_paths") or [])
    additions = [
        {
            "file": "facts/facts_for_bot_FOTON.yaml",
            "path": "schedule_2026_27.groups",
            "reason": "Foton 2026/27 schedule facts must come from YAML.",
        },
        {
            "file": "facts/facts_for_bot_UNPK.yaml",
            "path": "schedule_2026_27.groups",
            "reason": "UNPK 2026/27 schedule facts must come from YAML.",
        },
    ]
    existing = {(str(item.get("file")), str(item.get("path"))) for item in required_paths if isinstance(item, Mapping)}
    for item in additions:
        key = (item["file"], item["path"])
        if key not in existing:
            required_paths.append(item)
    manifest["required_yaml_paths"] = required_paths
    path.write_text(yaml.safe_dump(manifest, allow_unicode=True, sort_keys=False), encoding="utf-8")


def write_ingest_decision(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    fields = [
        "group_id",
        "status",
        "decision",
        "brand",
        "format",
        "place",
        "day",
        "time",
        "subject",
        "grade",
        "track",
        "date_start",
        "date_finish",
        "sheet_cell",
        "match_key",
        "note",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_derivation_reports(
    *,
    audit_pack: Path,
    schedule_pack: Path,
    source_dir: Path,
    release_id: str,
    freshness_date: str,
    summary: Mapping[str, Any],
    ingest_rows: Sequence[Mapping[str, str]],
    facts_by_brand: Mapping[str, Mapping[str, Any]],
    examples: Mapping[str, Sequence[str]],
    empty_filial_group_ids: Sequence[str],
) -> None:
    counts_by_brand = {brand: len(items) for brand, items in facts_by_brand.items()}
    status_counts = Counter(row.get("status", "") for row in ingest_rows)
    decision_counts = Counter(row.get("decision", "") for row in ingest_rows)
    report = {
        "schema_version": "kb_schedule_2026_27_derivation_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "release_id": release_id,
        "freshness_date": freshness_date,
        "schedule_pack": str(schedule_pack),
        "source_dir": str(source_dir),
        "input_summary": dict(summary),
        "status_counts": dict(status_counts),
        "decision_counts": dict(decision_counts),
        "facts_by_brand": counts_by_brand,
        "empty_filial_fallback_group_ids": list(empty_filial_group_ids),
        "client_safe_examples": {brand: list(items[:3]) for brand, items in examples.items()},
    }
    (audit_pack / "schedule_derivation_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    lines = [
        "# Деривация фактов расписания 2026/27",
        "",
        f"- Новый release_id: `{release_id}`.",
        f"- Источник сверки: `{schedule_pack}`.",
        f"- YAML-источники: `{source_dir}`.",
        f"- Дата свежести фактов: `{freshness_date}`.",
        f"- Входной gate: matched `{summary.get('matched')}`, mismatched `{summary.get('mismatched_field')}`, missing `{summary.get('missing_in_tallanto')}`, duplicate `{summary.get('duplicate_tallanto_matches')}`, extra `{summary.get('extra_in_tallanto')}`.",
        f"- Решения ingest: `{dict(decision_counts)}`.",
        f"- Фактов Фотон: `{counts_by_brand.get('foton', 0)}`.",
        f"- Фактов УНПК: `{counts_by_brand.get('unpk', 0)}`.",
        f"- Группы с fallback по пустому филиалу Tallanto: `{', '.join(empty_filial_group_ids) or 'нет'}`.",
        "",
        "## Примеры client_safe",
        "",
    ]
    for brand, brand_examples in examples.items():
        lines.append(f"### {brand}")
        for text in brand_examples[:3]:
            lines.append(f"- {text}")
        lines.append("")
    (audit_pack / "schedule_derivation_summary.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def safe_slug(value: str) -> str:
    text = value.strip().casefold()
    text = text.replace("ё", "e")
    translit = {
        "а": "a",
        "б": "b",
        "в": "v",
        "г": "g",
        "д": "d",
        "е": "e",
        "ж": "zh",
        "з": "z",
        "и": "i",
        "й": "j",
        "к": "k",
        "л": "l",
        "м": "m",
        "н": "n",
        "о": "o",
        "п": "p",
        "р": "r",
        "с": "s",
        "т": "t",
        "у": "u",
        "ф": "f",
        "х": "h",
        "ц": "c",
        "ч": "ch",
        "ш": "sh",
        "щ": "sch",
        "ъ": "",
        "ы": "y",
        "ь": "",
        "э": "e",
        "ю": "yu",
        "я": "ya",
    }
    text = "".join(translit.get(char, char) for char in text)
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def time_slug(value: str) -> str:
    numbers = re.findall(r"\d{1,2}:\d{2}", value)
    if not numbers:
        return safe_slug(value)
    return "_".join(part.replace(":", "") for part in numbers)


def format_date_ru(value: str) -> str:
    match = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", value.strip())
    if not match:
        return value
    year, month, day = match.groups()
    return f"{day}.{month}.{year}"


def capitalize_first(value: str) -> str:
    if not value:
        return value
    return value[:1].upper() + value[1:]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise DerivationError(f"Expected mapping in {path}.")
    return payload


def as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


if __name__ == "__main__":
    raise SystemExit(main())
