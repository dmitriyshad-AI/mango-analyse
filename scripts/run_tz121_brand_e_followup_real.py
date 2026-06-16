#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from mango_mvp.customer_timeline.canonical_readonly_import import infer_brand, infer_offline_brand


DEFAULT_REVIEW = "/Users/dmitrijfabarisov/Projects/Mango_tz116_offline/audits/_inbox/tz116_followup_gold_reviews_20260615/e_brand_loss_gold_sample.csv"
DEFAULT_MASTER = "/Users/dmitrijfabarisov/Projects/Mango analyse/stable_runtime/sales_master_export_20260523_audio_working_store_v1/master_contacts_ru.csv"


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    review_path = Path(args.review).expanduser().resolve()
    master_path = Path(args.master_contacts).expanduser().resolve()
    review_rows = read_csv(review_path)
    master_rows = read_master_rows(master_path, {int(row["row_index"]) for row in review_rows})
    trace_rows = [
        build_trace_row(row, master_rows[int(row["row_index"])], include_fragments=bool(args.include_fragments))
        for row in review_rows
    ]
    summary = build_summary(trace_rows, review_path=review_path, master_path=master_path, out_dir=out_dir)

    write_csv(out_dir / "tz121_e_brand_followup_trace.csv", trace_rows)
    write_jsonl(out_dir / "tz121_e_brand_followup_trace.jsonl", trace_rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "REPORT.md").write_text(render_report(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["gate_passed"] else 2


def build_trace_row(review: Mapping[str, Any], master_row: Mapping[str, Any], *, include_fragments: bool) -> dict[str, Any]:
    row_index = int(review["row_index"])
    legacy = infer_brand(master_row.values(), mode="legacy")
    flat_v2 = infer_brand(master_row.values(), mode="cyrillic_v2")
    offline_primary = infer_offline_brand(master_row)
    verdict = str(review.get("verdict") or "")
    flip = str(review.get("flip") or "")
    expected = expected_brand(verdict=verdict, flip=flip)
    return {
        "id": f"master_contacts:{row_index}",
        "input_fragment": input_fragment(master_row, include_fragments=include_fragments),
        "gold": expected,
        "rule": legacy,
        "model": offline_primary,
        "flat_cyrillic_v2": flat_v2,
        "confidence": "1.000000",
        "rationale": rationale(verdict=verdict, flip=flip, offline_primary=offline_primary),
        "matched_gold": "Да" if offline_primary == expected else "Нет",
        "error_type": classify_error_type(gold=expected, rule=legacy, model=offline_primary),
        "review_verdict": verdict,
        "review_flip": flip,
        "review_reason": review.get("reason", ""),
    }


def expected_brand(*, verdict: str, flip: str) -> str:
    if verdict == "false_negative" and flip == "foton->unknown":
        return "foton"
    if verdict == "false_negative" and flip == "unpk->unknown":
        return "unpk"
    return "unknown"


def input_fragment(master_row: Mapping[str, Any], *, include_fragments: bool) -> str:
    if not include_fragments:
        return "real master_contacts row redacted; see local ignored source"
    values = [str(value).replace("\n", " ") for value in master_row.values() if value]
    return " | ".join(values)[:240]


def rationale(*, verdict: str, flip: str, offline_primary: str) -> str:
    if verdict == "false_negative" and flip == "foton->unknown" and offline_primary == "foton":
        return "Полевая cyrillic_v2-логика восстановила явный Фотон без ослабления expected fail-closed."
    if verdict == "expected_fail_closed" and offline_primary == "unknown":
        return "Случай оставлен fail-closed: смешение брендов или небрендовая форма Фотона."
    if verdict == "unclear":
        return "Спорный gold-кейс не используется как разрешение на primary."
    return "Результат сверяется с ручным follow-up gold."


def classify_error_type(*, gold: str, rule: str, model: str) -> str:
    rule_ok = rule == gold
    model_ok = model == gold
    if model_ok and not rule_ok:
        return "model_fix"
    if rule_ok and not model_ok:
        return "model_break"
    if model_ok and rule_ok:
        return "both_correct"
    if rule == model:
        return "both_wrong_same"
    return "both_wrong"


def build_summary(
    rows: list[Mapping[str, Any]],
    *,
    review_path: Path,
    master_path: Path,
    out_dir: Path,
) -> dict[str, Any]:
    verdict_counts = Counter(str(row["review_verdict"]) for row in rows)
    error_counts = Counter(str(row["error_type"]) for row in rows)
    foton_false_negative_rows = [
        row for row in rows if row["review_flip"] == "foton->unknown" and row["review_verdict"] == "false_negative"
    ]
    expected_fail_closed_rows = [row for row in rows if row["review_verdict"] == "expected_fail_closed"]
    gate_passed = all(row["model"] == "foton" for row in foton_false_negative_rows) and all(
        row["model"] == "unknown" for row in expected_fail_closed_rows
    )
    return {
        "schema_version": "tz121_e_brand_followup_real_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "primary_followup_gate",
        "review_input": str(review_path),
        "master_contacts": str(master_path),
        "out_dir": str(out_dir),
        "rows_total": len(rows),
        "verdict_counts": dict(verdict_counts.most_common()),
        "foton_false_negative_rows": len(foton_false_negative_rows),
        "foton_false_negative_fixed": sum(1 for row in foton_false_negative_rows if row["model"] == "foton"),
        "expected_fail_closed_rows": len(expected_fail_closed_rows),
        "expected_fail_closed_kept_unknown": sum(1 for row in expected_fail_closed_rows if row["model"] == "unknown"),
        "model_break_rows": sum(1 for row in rows if row["error_type"] == "model_break"),
        "error_type_counts": dict(error_counts.most_common()),
        "gate_passed": gate_passed,
        "llm_calls_total": 0,
        "safety": {
            "calls_model": False,
            "uses_openai_api_key": False,
            "writes_db": False,
            "writes_crm": False,
            "writes_tallanto": False,
            "reads_stable_runtime": True,
            "writes_raw_pii_to_git": False,
            "fragments_redacted_by_default": True,
        },
    }


def render_report(summary: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# TZ-121 E Follow-up Gate",
            "",
            f"- Mode: `{summary['mode']}`",
            f"- Rows: `{summary['rows_total']}`",
            f"- Foton false-negative fixed: `{summary['foton_false_negative_fixed']}/{summary['foton_false_negative_rows']}`",
            f"- Expected fail-closed kept unknown: `{summary['expected_fail_closed_kept_unknown']}/{summary['expected_fail_closed_rows']}`",
            f"- Model breaks: `{summary['model_break_rows']}`",
            f"- Gate passed: `{summary['gate_passed']}`",
            f"- Error types: `{json.dumps(summary['error_type_counts'], ensure_ascii=False, sort_keys=True)}`",
            "",
            "Safety: read-only local master_contacts, no model calls, no DB/CRM/Tallanto writes; raw fragments are redacted by default.",
        ]
    ) + "\n"


def read_master_rows(path: Path, row_indexes: set[int]) -> dict[int, dict[str, str]]:
    rows: dict[int, dict[str, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row_number, row in enumerate(reader, start=2):
            if row_number in row_indexes:
                rows[row_number] = row
    missing = sorted(row_indexes - set(rows))
    if missing:
        raise SystemExit(f"Missing master_contacts rows: {missing}")
    return rows


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    fields = list(rows[0]) if rows else ["empty"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TZ-121 E: real follow-up gate before cyrillic_v2 primary.")
    parser.add_argument("--review", default=DEFAULT_REVIEW)
    parser.add_argument("--master-contacts", default=DEFAULT_MASTER)
    parser.add_argument("--out-dir", default="audits/_inbox/tz121_e_brand_followup_real")
    parser.add_argument("--include-fragments", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
