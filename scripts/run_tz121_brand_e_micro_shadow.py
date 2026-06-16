#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from mango_mvp.customer_timeline.canonical_readonly_import import infer_brand


DEFAULT_INPUT = "tests/fixtures/tz121_brand_e_micro_gold.csv"


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    mode = str(args.mode or "shadow").strip().lower()
    if mode != "shadow":
        raise SystemExit("TZ-121 E micro runner is shadow-only until Claude/Dmitry regrede.")

    input_path = Path(args.input).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [build_trace_row(row, index=index) for index, row in enumerate(read_csv(input_path), start=1)]
    summary = build_summary(rows, input_path=input_path, out_dir=out_dir)

    write_csv(out_dir / "tz121_e_brand_trace.csv", rows)
    write_jsonl(out_dir / "tz121_e_brand_trace.jsonl", rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "REPORT.md").write_text(render_report(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def build_trace_row(row: Mapping[str, Any], *, index: int) -> dict[str, Any]:
    case_id = str(row.get("case_id") or f"row:{index}")
    text = str(row.get("input_text") or "")
    gold = str(row.get("gold_brand") or "unknown").strip().lower()
    legacy = infer_brand([text], mode="legacy")
    model = infer_brand([text], mode="cyrillic_v2")
    return {
        "id": case_id,
        "input_fragment": text[:240],
        "gold": gold,
        "rule": legacy,
        "model": model,
        "confidence": "1.000000",
        "rationale": rationale_for(text=text, legacy=legacy, model=model),
        "matched_gold": "Да" if model == gold else "Нет",
        "error_type": classify_error_type(gold=gold, rule=legacy, model=model),
        "flip": f"{legacy}->{model}",
        "case_note": row.get("case_note", ""),
    }


def rationale_for(*, text: str, legacy: str, model: str) -> str:
    lowered = text.casefold().replace("ё", "е")
    compact = "".join(lowered.split())
    has_foton = "фотон" in compact or "foton" in compact
    has_unpk = "унпк" in compact or "unpk" in compact or "мфти" in compact
    if has_foton and has_unpk:
        return "Найдены корни двух брендов; fail-closed в unknown."
    if model == "foton":
        return "Найден корень фотон с учетом склонений/склеек."
    if model == "unpk":
        return "Найден корень УНПК/МФТИ."
    if legacy == model:
        return "Бренд не найден; unknown сохраняется."
    return "cyrillic_v2 изменил бренд по корневому матчингу."


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


def build_summary(rows: list[Mapping[str, Any]], *, input_path: Path, out_dir: Path) -> dict[str, Any]:
    error_counts = Counter(str(row.get("error_type") or "") for row in rows)
    flips = Counter(str(row.get("flip") or "") for row in rows)
    gold_counts = Counter(str(row.get("gold") or "unknown") for row in rows)
    foton_rows = [row for row in rows if row.get("gold") == "foton"]
    cross_rows = [row for row in rows if row.get("gold") == "unknown" and row.get("rule") != "unknown"]
    return {
        "schema_version": "tz121_e_brand_micro_shadow_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "shadow",
        "input": str(input_path),
        "out_dir": str(out_dir),
        "rows_total": len(rows),
        "gold_counts": dict(gold_counts.most_common()),
        "model_correct": sum(1 for row in rows if row.get("matched_gold") == "Да"),
        "model_break_rows": sum(1 for row in rows if row.get("error_type") == "model_break"),
        "foton_gold_rows": len(foton_rows),
        "foton_unknown_legacy": sum(1 for row in foton_rows if row.get("rule") == "unknown"),
        "foton_unknown_cyrillic_v2": sum(1 for row in foton_rows if row.get("model") == "unknown"),
        "cross_brand_rows": len(cross_rows),
        "cross_brand_fail_closed": sum(1 for row in cross_rows if row.get("model") == "unknown"),
        "error_type_counts": dict(error_counts.most_common()),
        "flip_counts": dict(flips.most_common()),
        "llm_calls_total": 0,
        "primary_run": False,
        "stop_for_regrede": True,
        "safety": {
            "calls_model": False,
            "uses_openai_api_key": False,
            "writes_db": False,
            "writes_crm": False,
            "writes_tallanto": False,
            "reads_stable_runtime": False,
            "runs_full_set": False,
            "unknown_fail_closed": True,
        },
    }


def render_report(summary: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# TZ-121 E Brand Micro Shadow",
            "",
            f"- Mode: `{summary['mode']}`",
            f"- Rows: `{summary['rows_total']}`",
            f"- Model correct: `{summary['model_correct']}`",
            f"- Model breaks: `{summary['model_break_rows']}`",
            f"- Foton gold rows: `{summary['foton_gold_rows']}`",
            f"- Foton unknown legacy: `{summary['foton_unknown_legacy']}`",
            f"- Foton unknown cyrillic_v2: `{summary['foton_unknown_cyrillic_v2']}`",
            f"- Cross-brand rows: `{summary['cross_brand_rows']}`",
            f"- Cross-brand fail-closed: `{summary['cross_brand_fail_closed']}`",
            f"- Error types: `{json.dumps(summary['error_type_counts'], ensure_ascii=False, sort_keys=True)}`",
            f"- Flips: `{json.dumps(summary['flip_counts'], ensure_ascii=False, sort_keys=True)}`",
            "",
            "Safety: synthetic micro-set only, no model calls, no full set, no DB/CRM/Tallanto writes.",
            "",
            "Stop: wait for Claude/Dmitry regrede before enabling E primary or moving to C.",
        ]
    ) + "\n"


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
    parser = argparse.ArgumentParser(description="TZ-121 E: shadow micro-measurement for brand root matcher.")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", default="audits/_inbox/tz121_e_brand_micro_shadow")
    parser.add_argument("--mode", default="shadow")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
