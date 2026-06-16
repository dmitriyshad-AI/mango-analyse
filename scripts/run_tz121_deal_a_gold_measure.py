#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_RESULTS = (
    "/Users/dmitrijfabarisov/Projects/Mango_tz116_offline/audits/_inbox/"
    "tz116_crm_llm_shadow_fixed24_codex_20260615_195654/crm_llm_offline_measure_results.jsonl"
)
VALID_VERDICTS = {"closed_valid", "closed_too_early", "follow_up_needed", "manual_review"}
VALID_RISKS = {"no_risk", "low", "medium", "high", "manual_review"}
VALID_NEXT_STEP_CLASSES = {"follow_up_check", "manual_check", "no_action", "other"}


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    mode = normalize_mode(args.mode)
    if mode != "shadow":
        raise SystemExit("TZ-121 A runner is shadow-only until Claude/Dmitry regrede.")

    results_path = Path(args.results).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    records = load_json_records(results_path)
    if not records:
        raise SystemExit(f"No records loaded from {results_path}")

    gold_path = Path(args.gold).expanduser().resolve() if args.gold else out_dir / "deal_a_gold_labels.csv"
    if args.write_gold or not gold_path.exists():
        gold_rows = build_conservative_gold(records)
        write_csv(gold_path, gold_rows)
    gold = read_gold(gold_path)

    trace_rows = [
        build_trace_row(record, gold=gold, index=index, include_fragments=bool(args.include_fragments))
        for index, record in enumerate(records, start=1)
    ]
    summary = build_summary(trace_rows, results_path=results_path, gold_path=gold_path, out_dir=out_dir, mode=mode)

    write_csv(out_dir / "tz121_a_deal_gold_trace.csv", trace_rows)
    write_jsonl(out_dir / "tz121_a_deal_gold_trace.jsonl", trace_rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "REPORT.md").write_text(render_report(summary, trace_rows), encoding="utf-8")
    (out_dir / "semantic_review.md").write_text(render_semantic_review(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def build_conservative_gold(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        case_id = record_case_id(record, index)
        heuristic = analysis(record, "heuristic_analysis")
        model = analysis(record, "llm_analysis")
        selected = conservative_gold_label(heuristic=heuristic, model=model)
        rows.append(
            {
                "case_id": case_id,
                "brand": safe_str(heuristic.get("brand")),
                "gold_verdict": selected["verdict"],
                "gold_risk": selected["risk"],
                "gold_next_step_class": selected["next_step_class"],
                "gold_reason": selected["reason"],
                "review_policy": "tz121_a_conservative_manual_v1",
            }
        )
    return rows


def conservative_gold_label(*, heuristic: Mapping[str, Any], model: Mapping[str, Any]) -> dict[str, str]:
    heuristic_verdict = safe_str(heuristic.get("close_verdict"))
    model_verdict = safe_str(model.get("close_verdict"))
    heuristic_risk = safe_str(heuristic.get("premature_close_risk"))
    model_risk = safe_str(model.get("premature_close_risk"))
    loss = normalize_text(heuristic.get("loss_reason_summary") or model.get("loss_reason_summary"))

    if heuristic_verdict == model_verdict and heuristic_risk == model_risk:
        return {
            "verdict": heuristic_verdict,
            "risk": heuristic_risk,
            "next_step_class": next_step_class_for_verdict(heuristic_verdict),
            "reason": "rule_and_model_agree",
        }

    if "архив" in loss and "связ" in loss:
        return {
            "verdict": "manual_review",
            "risk": "manual_review",
            "next_step_class": "manual_check",
            "reason": "archive_no_contact_needs_manual_review_before_auto_conclusion",
        }

    if model_verdict == "manual_review" or model_risk == "manual_review":
        return {
            "verdict": "manual_review",
            "risk": "manual_review",
            "next_step_class": "manual_check",
            "reason": "model_requests_manual_review_on_disagreement",
        }

    return {
        "verdict": heuristic_verdict or "manual_review",
        "risk": heuristic_risk or "manual_review",
        "next_step_class": next_step_class_for_verdict(heuristic_verdict),
        "reason": "fallback_to_legacy_on_unresolved_disagreement",
    }


def build_trace_row(
    record: Mapping[str, Any],
    *,
    gold: Mapping[str, Mapping[str, str]],
    index: int,
    include_fragments: bool,
) -> dict[str, Any]:
    case_id = record_case_id(record, index)
    heuristic = analysis(record, "heuristic_analysis")
    model = analysis(record, "llm_analysis")
    gold_row = gold.get(case_id)
    if not gold_row:
        raise SystemExit(f"Missing gold label for {case_id}")

    rule_verdict = safe_str(heuristic.get("close_verdict"))
    rule_risk = safe_str(heuristic.get("premature_close_risk"))
    rule_next = next_step_class_for_verdict(rule_verdict)
    model_verdict = safe_str(model.get("close_verdict"))
    model_risk = safe_str(model.get("premature_close_risk"))
    model_next = next_step_class_for_verdict(model_verdict)

    gold_verdict = gold_row["gold_verdict"]
    gold_risk = gold_row["gold_risk"]
    gold_next = gold_row["gold_next_step_class"]
    rule_exact = exact_match(
        verdict=rule_verdict,
        risk=rule_risk,
        next_step_class=rule_next,
        gold_verdict=gold_verdict,
        gold_risk=gold_risk,
        gold_next_step_class=gold_next,
    )
    model_exact = exact_match(
        verdict=model_verdict,
        risk=model_risk,
        next_step_class=model_next,
        gold_verdict=gold_verdict,
        gold_risk=gold_risk,
        gold_next_step_class=gold_next,
    )
    confidence = safe_float(model.get("confidence"))
    return {
        "id": case_id,
        "input_fragment": input_fragment(heuristic, model, include_fragments=include_fragments),
        "brand": safe_str(heuristic.get("brand") or record.get("brand")),
        "gold_verdict": gold_verdict,
        "gold_risk": gold_risk,
        "gold_next_step_class": gold_next,
        "rule_verdict": rule_verdict,
        "rule_risk": rule_risk,
        "rule_next_step_class": rule_next,
        "model_verdict": model_verdict,
        "model_risk": model_risk,
        "model_next_step_class": model_next,
        "confidence": f"{confidence:.6f}",
        "rationale": model_rationale(model),
        "gold_reason": gold_row.get("gold_reason", ""),
        "rule_matches_gold": "Да" if rule_exact else "Нет",
        "model_matches_gold": "Да" if model_exact else "Нет",
        "error_type": classify_error_type(rule_ok=rule_exact, model_ok=model_exact),
        "verdict_error_type": classify_error_type(
            rule_ok=rule_verdict == gold_verdict,
            model_ok=model_verdict == gold_verdict,
        ),
        "risk_error_type": classify_error_type(rule_ok=rule_risk == gold_risk, model_ok=model_risk == gold_risk),
        "next_step_error_type": classify_error_type(rule_ok=rule_next == gold_next, model_ok=model_next == gold_next),
        "writeback_allowed": "Нет",
        "classification_method": "tz121_a_deal_gold_shadow",
    }


def build_summary(
    rows: list[Mapping[str, Any]],
    *,
    results_path: Path,
    gold_path: Path,
    out_dir: Path,
    mode: str,
) -> dict[str, Any]:
    brands = Counter(safe_str(row.get("brand")) for row in rows)
    error_counts = Counter(safe_str(row.get("error_type")) for row in rows)
    verdict_errors = Counter(safe_str(row.get("verdict_error_type")) for row in rows)
    risk_errors = Counter(safe_str(row.get("risk_error_type")) for row in rows)
    next_step_errors = Counter(safe_str(row.get("next_step_error_type")) for row in rows)
    high_conf_wrong = [
        row
        for row in rows
        if row.get("model_matches_gold") == "Нет" and safe_float(row.get("confidence")) >= 0.8
    ]
    model_correct = sum(1 for row in rows if row.get("model_matches_gold") == "Да")
    rule_correct = sum(1 for row in rows if row.get("rule_matches_gold") == "Да")
    return {
        "schema_version": "tz121_a_deal_gold_shadow_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "input": str(results_path),
        "gold": str(gold_path),
        "out_dir": str(out_dir),
        "records_total": len(rows),
        "brand_counts": dict(brands.most_common()),
        "rule_exact_vs_gold": {
            "correct": rule_correct,
            "total": len(rows),
            "accuracy": rule_correct / len(rows) if rows else 0.0,
        },
        "model_exact_vs_gold": {
            "correct": model_correct,
            "total": len(rows),
            "accuracy": model_correct / len(rows) if rows else 0.0,
        },
        "model_delta_vs_rule": model_correct - rule_correct,
        "error_type_counts": dict(error_counts.most_common()),
        "verdict_error_type_counts": dict(verdict_errors.most_common()),
        "risk_error_type_counts": dict(risk_errors.most_common()),
        "next_step_error_type_counts": dict(next_step_errors.most_common()),
        "high_confidence_wrong_count": len(high_conf_wrong),
        "llm_calls_total": 0,
        "primary_run": False,
        "stop_for_regrede": True,
        "safety": {
            "calls_model": False,
            "uses_openai_api_key": False,
            "uses_precomputed_codex_shadow": True,
            "writes_db": False,
            "writes_crm": False,
            "writes_amo": False,
            "writes_tallanto": False,
            "reads_live_crm": False,
            "writeback_allowed": False,
            "raw_pii_written_to_git": False,
        },
    }


def render_report(summary: Mapping[str, Any], rows: list[Mapping[str, Any]]) -> str:
    disagreements = [row for row in rows if row["error_type"] != "both_correct"][:15]
    return "\n".join(
        [
            "# TZ-121 A Deal Gold Shadow",
            "",
            f"- Mode: `{summary['mode']}`",
            f"- Rows: `{summary['records_total']}`",
            f"- Brand split: `{json.dumps(summary['brand_counts'], ensure_ascii=False, sort_keys=True)}`",
            f"- Rule exact vs gold: `{summary['rule_exact_vs_gold']['correct']}/{summary['rule_exact_vs_gold']['total']}` = `{summary['rule_exact_vs_gold']['accuracy']:.4f}`",
            f"- Model exact vs gold: `{summary['model_exact_vs_gold']['correct']}/{summary['model_exact_vs_gold']['total']}` = `{summary['model_exact_vs_gold']['accuracy']:.4f}`",
            f"- Model delta vs rule: `{summary['model_delta_vs_rule']}`",
            f"- High-confidence wrong model rows: `{summary['high_confidence_wrong_count']}`",
            f"- Error types: `{json.dumps(summary['error_type_counts'], ensure_ascii=False, sort_keys=True)}`",
            f"- LLM calls total in this run: `{summary['llm_calls_total']}`",
            "",
            "Safety: uses saved Codex shadow results only, no live CRM reads, no model calls, no DB/AMO/Tallanto writes.",
            "",
            "## First Non-Both-Correct Rows",
            "",
            "| brand | gold | rule | model | confidence | type | gold reason |",
            "|---|---|---|---|---:|---|---|",
            *[
                "| `{brand}` | `{gold}` | `{rule}` | `{model}` | `{conf}` | `{err}` | `{reason}` |".format(
                    brand=row["brand"],
                    gold="/".join([row["gold_verdict"], row["gold_risk"], row["gold_next_step_class"]]),
                    rule="/".join([row["rule_verdict"], row["rule_risk"], row["rule_next_step_class"]]),
                    model="/".join([row["model_verdict"], row["model_risk"], row["model_next_step_class"]]),
                    conf=row["confidence"],
                    err=row["error_type"],
                    reason=row["gold_reason"],
                )
                for row in disagreements
            ],
            "",
            "Stop: A primary is not enabled. Claude/Dmitry regrede decides whether a primary path is safe.",
        ]
    ) + "\n"


def render_semantic_review(summary: Mapping[str, Any]) -> str:
    verdict = "PASS_WITH_NOTES"
    return "\n".join(
        [
            "# Semantic Review: TZ-121 A Deal Gold Shadow",
            "",
            f"Verdict: `{verdict}`",
            "",
            "## What Passed",
            "",
            "- The measurement stays offline and read-only.",
            "- The gold policy is conservative: uncertain archive/no-contact cases require manual review before any automatic conclusion.",
            "- No writeback path is enabled.",
            "",
            "## Non-Blocking Risks",
            "",
            "- The 24-case gold set is intentionally small and should be independently regreded before primary.",
            "- Case identifiers remain only in ignored audit artifacts; the tracked report uses aggregate counters.",
            "",
            "## Required Gate",
            "",
            "- A primary decision for deal analysis must require Claude/Dmitry regrede on the raw ignored trace.",
            "",
            f"Model exact vs gold: `{summary['model_exact_vs_gold']['correct']}/{summary['model_exact_vs_gold']['total']}`.",
            f"Rule exact vs gold: `{summary['rule_exact_vs_gold']['correct']}/{summary['rule_exact_vs_gold']['total']}`.",
        ]
    ) + "\n"


def exact_match(
    *,
    verdict: str,
    risk: str,
    next_step_class: str,
    gold_verdict: str,
    gold_risk: str,
    gold_next_step_class: str,
) -> bool:
    return verdict == gold_verdict and risk == gold_risk and next_step_class == gold_next_step_class


def classify_error_type(*, rule_ok: bool, model_ok: bool) -> str:
    if model_ok and not rule_ok:
        return "model_fix"
    if rule_ok and not model_ok:
        return "model_break"
    if model_ok and rule_ok:
        return "both_correct"
    return "both_wrong"


def model_rationale(model: Mapping[str, Any]) -> str:
    signals = model.get("evidence_signals")
    if isinstance(signals, list) and signals:
        return safe_str(signals[0])[:240]
    flags = model.get("conflict_flags")
    if isinstance(flags, list) and flags:
        return safe_str(flags[0])[:240]
    summary = model.get("close_reason_summary") or model.get("deal_summary")
    return safe_str(summary)[:240]


def input_fragment(heuristic: Mapping[str, Any], model: Mapping[str, Any], *, include_fragments: bool) -> str:
    if not include_fragments:
        return "redacted closed-deal dossier"
    return "; ".join(
        [
            f"brand={safe_str(heuristic.get('brand'))}",
            f"pipeline={safe_str(heuristic.get('pipeline_name'))}",
            f"status={safe_str(heuristic.get('status_name'))}",
            f"loss={safe_str(heuristic.get('loss_reason_summary') or model.get('loss_reason_summary'))}",
        ]
    )[:300]


def next_step_class_for_verdict(verdict: str) -> str:
    if verdict == "closed_valid":
        return "no_action"
    if verdict in {"follow_up_needed", "closed_too_early"}:
        return "follow_up_check"
    if verdict == "manual_review":
        return "manual_check"
    return "other"


def read_gold(path: Path) -> dict[str, dict[str, str]]:
    rows = read_csv(path)
    gold: dict[str, dict[str, str]] = {}
    for row in rows:
        case_id = safe_str(row.get("case_id"))
        if not case_id:
            raise SystemExit(f"Gold row without case_id in {path}")
        verdict = safe_str(row.get("gold_verdict"))
        risk = safe_str(row.get("gold_risk"))
        next_step = safe_str(row.get("gold_next_step_class"))
        if verdict not in VALID_VERDICTS:
            raise SystemExit(f"Invalid gold_verdict for {case_id}: {verdict}")
        if risk not in VALID_RISKS:
            raise SystemExit(f"Invalid gold_risk for {case_id}: {risk}")
        if next_step not in VALID_NEXT_STEP_CLASSES:
            raise SystemExit(f"Invalid gold_next_step_class for {case_id}: {next_step}")
        gold[case_id] = row
    return gold


def load_json_records(path: Path) -> list[dict[str, Any]]:
    text = read_text(path)
    records: list[dict[str, Any]] = []
    for line in text.split("\n"):
        if not line.strip():
            continue
        records.append(json.loads(line))
    return records


def read_text(path: Path) -> str:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return handle.read()
    return path.read_text(encoding="utf-8")


def analysis(record: Mapping[str, Any], field: str) -> Mapping[str, Any]:
    value = record.get(field)
    if not isinstance(value, Mapping):
        raise SystemExit(f"Record {record.get('case_id')} missing {field}")
    return value


def record_case_id(record: Mapping[str, Any], index: int) -> str:
    return safe_str(record.get("case_id") or f"row:{index}")


def normalize_text(value: Any) -> str:
    return safe_str(value).replace("\u2028", " ").lower().replace("ё", "е")


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def safe_str(value: Any) -> str:
    return "" if value is None else str(value).strip()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    fields = list(rows[0]) if rows else ["empty"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def normalize_mode(mode: str) -> str:
    normalized = str(mode or "shadow").strip().lower()
    if normalized not in {"shadow"}:
        raise SystemExit("TZ-121 A runner is shadow-only until Claude/Dmitry regrede.")
    return normalized


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TZ-121 A: gold measurement for offline deal analysis.")
    parser.add_argument("--results", default=DEFAULT_RESULTS)
    parser.add_argument("--gold", default="")
    parser.add_argument("--out-dir", default="audits/_inbox/tz121_a_deal_gold_shadow")
    parser.add_argument("--mode", default="shadow")
    parser.add_argument("--write-gold", action="store_true")
    parser.add_argument("--include-fragments", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
