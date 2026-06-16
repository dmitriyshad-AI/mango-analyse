#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


TRACE_FIELDS = [
    "id",
    "call_id",
    "turn_index",
    "input_fragment",
    "gold",
    "rule",
    "raw_model",
    "model",
    "confidence",
    "rationale",
    "is_guarded",
    "guard_changed",
    "raw_matched_gold",
    "post_matched_gold",
    "guard_effect_type",
    "matched_gold",
    "error_type",
]

DEFAULT_C = Path("audits/_inbox/tz116_question_catalog_labeled100_codex_shadow_20260615_192755")
DEFAULT_A = Path("audits/_inbox/tz116_crm_llm_shadow_fixed24_codex_20260615_195654")
DEFAULT_D = Path("audits/_inbox/tz116_mono_role_gold23_rerun_20260615_221929")
DEFAULT_GOLD = Path("audits/_inbox/tz116_followup_gold_reviews_20260615")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    requested_blocks = parse_blocks(args.blocks)
    trace_builders = {
        "c": lambda: build_c_trace(Path(args.c_dir)),
        "d": lambda: build_d_trace(Path(args.d_dir)),
        "a": lambda: build_a_trace(Path(args.a_dir)),
        "b": lambda: build_review_trace(Path(args.gold_dir) / "b_outcome_flip_gold_sample.csv", block="b"),
        "e": lambda: build_review_trace(Path(args.gold_dir) / "e_brand_loss_gold_sample.csv", block="e"),
    }
    traces = {block: trace_builders[block]() for block in requested_blocks}

    summaries: dict[str, Any] = {}
    for block, rows in traces.items():
        block_dir = out_dir / block
        block_dir.mkdir(parents=True, exist_ok=True)
        write_csv(block_dir / f"{block}_trace.csv", rows)
        write_jsonl(block_dir / f"{block}_trace.jsonl", rows)
        summary = summarize_trace(rows)
        summaries[block] = summary
        (block_dir / f"{block}_trace_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (block_dir / f"{block}_trace_REPORT.md").write_text(
            render_trace_report(block, summary, rows),
            encoding="utf-8",
        )

    combined = {
        "schema_version": "tz117_error_trace_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "out_dir": str(out_dir),
        "blocks": summaries,
        "safety": {
            "uses_codex_cli_only": True,
            "calls_model": False,
            "uses_openai_api_key": False,
            "writes_crm": False,
            "writes_tallanto": False,
            "writes_stable_runtime": False,
            "changes_block_behavior": False,
        },
    }
    (out_dir / "tz117_trace_summary.json").write_text(
        json.dumps(combined, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "TZ117_TRACE_REPORT.md").write_text(render_combined_report(combined), encoding="utf-8")
    print(json.dumps(combined, ensure_ascii=False, indent=2))
    return 0


def build_c_trace(c_dir: Path) -> list[dict[str, str]]:
    predictions = read_csv(c_dir / "question_catalog_offline_predictions.csv")
    reasoning_by_id: dict[str, str] = {}
    confidence_by_id: dict[str, str] = {}
    for item in read_jsonl(c_dir / "question_catalog_codex_shadow_predictions.jsonl"):
        item_id = str(item.get("question_item_id") or "")
        reasoning_by_id[item_id] = one_line(item.get("reasoning"))
        confidence_by_id[item_id] = str(item.get("confidence") or "")

    rows: list[dict[str, str]] = []
    for row in predictions:
        item_id = str(row.get("question_id") or row.get("question_id_resolved") or "")
        gold = str(row.get("human_label") or "")
        rule = str(row.get("rule_theme_id") or "")
        model = str(row.get("model_theme_id") or "")
        matched = bool(gold and model == gold)
        rows.append(
            {
                "id": item_id,
                "input_fragment": redact_fragment(row.get("raw_text"), limit=240),
                "gold": gold,
                "rule": rule,
                "model": model,
                "confidence": confidence_by_id.get(item_id) or str(row.get("model_confidence") or ""),
                "rationale": reasoning_by_id.get(item_id, ""),
                "matched_gold": bool_cell(matched),
                "error_type": classify_error_type(gold=gold, rule=rule, model=model),
            }
        )
    return rows


def build_a_trace(a_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for item in read_jsonl(a_dir / "crm_llm_offline_measure_results.jsonl"):
        case_id = str(item.get("case_id") or "")
        heuristic = item.get("heuristic_analysis") if isinstance(item.get("heuristic_analysis"), dict) else {}
        llm = item.get("llm_analysis") if isinstance(item.get("llm_analysis"), dict) else {}
        rule = combine_verdict(heuristic)
        model = combine_verdict(llm)
        confidence = str(llm.get("confidence") or "")
        rationale_parts = [llm.get("close_reason_summary")]
        evidence = llm.get("evidence_signals")
        if isinstance(evidence, list) and evidence:
            rationale_parts.append(evidence[0])
        rows.append(
            {
                "id": case_id,
                "input_fragment": redact_fragment(
                    " | ".join(
                        str(part or "")
                        for part in (
                            heuristic.get("brand"),
                            heuristic.get("pipeline_name"),
                            heuristic.get("status_name"),
                            heuristic.get("loss_reason_summary"),
                            heuristic.get("close_reason_summary"),
                        )
                    ),
                    limit=260,
                ),
                "gold": "",
                "rule": rule,
                "model": model,
                "confidence": confidence,
                "rationale": one_line(" | ".join(str(part or "") for part in rationale_parts if part)),
                "matched_gold": "",
                "error_type": "no_gold_agree" if rule == model else "no_gold_rule_model_disagree",
            }
        )
    return rows


def build_d_trace(d_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for item in read_jsonl(d_dir / "mono_role_gold50_measure_results.jsonl"):
        call_id = str(item.get("canonical_call_id") or "")
        turns = item.get("turns") if isinstance(item.get("turns"), list) else []
        gold_roles = list(item.get("gold_roles") or [])
        rule = item.get("rule") if isinstance(item.get("rule"), dict) else {}
        selected = item.get("selected") if isinstance(item.get("selected"), dict) else {}
        rule_roles = list(rule.get("roles") or [])
        model_roles = list(selected.get("roles") or [])
        selected_meta = selected.get("meta") if isinstance(selected.get("meta"), dict) else {}
        raw_model_roles = list(selected_meta.get("raw_model_roles_before_guard") or model_roles)
        confidence = str(selected.get("confidence") or selected_meta.get("confidence") or "")
        rationale = one_line(selected_meta.get("rationale") or selected_meta.get("notes"))
        low_info_policy = str(selected_meta.get("low_info_policy") or "")
        low_info_indexes = {
            int(value)
            for value in selected_meta.get("low_info_turn_indexes") or []
            if str(value).isdigit()
        }
        segment_guard_indexes = {
            int(value)
            for value in selected_meta.get("segment_guard_turn_indexes") or []
            if str(value).isdigit()
        }
        segment_guard_changed = {
            int(value)
            for value in selected_meta.get("segment_guard_changed_indexes") or []
            if str(value).isdigit()
        }
        segment_guard_reasons = (
            selected_meta.get("segment_guard_turn_reasons")
            if isinstance(selected_meta.get("segment_guard_turn_reasons"), dict)
            else {}
        )
        max_len = max(len(turns), len(gold_roles), len(rule_roles), len(model_roles), len(raw_model_roles))
        for index in range(max_len):
            gold = str(gold_roles[index]) if index < len(gold_roles) else ""
            rule_role = str(rule_roles[index]) if index < len(rule_roles) else ""
            raw_model = str(raw_model_roles[index]) if index < len(raw_model_roles) else ""
            model = str(model_roles[index]) if index < len(model_roles) else ""
            matched = bool(gold and model == gold)
            raw_matched = bool(gold and raw_model == gold)
            fragment = ""
            if index < len(turns) and isinstance(turns[index], Mapping):
                fragment = redact_fragment(turns[index].get("text"), limit=220)
            row_rationale = rationale
            if index + 1 in low_info_indexes:
                low_info_prefix = (
                    "low_info: короткая служебная реплика оставлена правилу"
                    if low_info_policy == "short_service_turns_keep_rule_role"
                    else "low_info: короткая служебная реплика помечена для ручной калибровки"
                )
                row_rationale = one_line(
                    low_info_prefix + " | " + rationale
                )
            guard_reason_raw = segment_guard_reasons.get(str(index + 1), [])
            if isinstance(guard_reason_raw, list) and guard_reason_raw:
                row_rationale = one_line(
                    "segment_guard: "
                    + ",".join(str(item) for item in guard_reason_raw)
                    + " | "
                    + row_rationale
                )
            guard_effect = guard_effect_type(
                gold=gold,
                raw_model=raw_model,
                post_model=model,
                is_guarded=index + 1 in segment_guard_indexes,
            )
            rows.append(
                {
                    "id": f"{call_id}:{index + 1}",
                    "call_id": call_id,
                    "turn_index": str(index + 1),
                    "input_fragment": fragment,
                    "gold": gold,
                    "rule": rule_role,
                    "raw_model": raw_model,
                    "model": model,
                    "confidence": confidence,
                    "rationale": row_rationale,
                    "is_guarded": bool_cell(index + 1 in segment_guard_indexes),
                    "guard_changed": bool_cell(index + 1 in segment_guard_changed),
                    "raw_matched_gold": bool_cell(raw_matched) if gold else "",
                    "post_matched_gold": bool_cell(matched) if gold else "",
                    "guard_effect_type": guard_effect,
                    "matched_gold": bool_cell(matched),
                    "error_type": classify_error_type(gold=gold, rule=rule_role, model=model),
                }
            )
    return rows


def build_review_trace(path: Path, *, block: str) -> list[dict[str, str]]:
    source = read_csv(path)
    rows: list[dict[str, str]] = []
    for row in source:
        row_id = str(row.get("row_index") or row.get("id") or "")
        flip = str(row.get("flip") or "")
        rule, model = split_flip(flip)
        verdict = str(row.get("verdict") or "")
        gold = review_gold_label(verdict, rule=rule, model=model, block=block)
        matched_gold = bool(gold and model == gold)
        rationale_parts = [row.get("reason"), f"review_verdict={verdict}" if verdict else ""]
        rows.append(
            {
                "id": f"{block}:{row_id}",
                "input_fragment": redact_fragment(row.get("reason"), limit=260),
                "gold": gold,
                "rule": rule,
                "model": model,
                "confidence": "1.0",
                "rationale": one_line(" | ".join(part for part in rationale_parts if part)),
                "matched_gold": bool_cell(matched_gold) if gold else "",
                "error_type": review_error_type(verdict, block=block, gold=gold, rule=rule, model=model),
            }
        )
    return rows


def summarize_trace(rows: list[dict[str, str]]) -> dict[str, Any]:
    type_counts = Counter(row["error_type"] for row in rows)
    guard_effect_counts = Counter(row.get("guard_effect_type") or "" for row in rows if row.get("guard_effect_type"))
    confidence_correct: list[float] = []
    confidence_errors: list[float] = []
    high_conf_wrong: list[dict[str, str]] = []
    confusion = Counter()
    gold_present_total = 0
    gold_absent_total = 0
    gold_unclear_total = 0
    for row in rows:
        confidence = parse_float(row.get("confidence"))
        matched_value = row.get("matched_gold")
        matched = matched_value == "Да"
        if row.get("gold"):
            gold_present_total += 1
            confusion[f"{row.get('gold')}->{row.get('model')}"] += 1
        elif row.get("error_type") == "gold_unclear":
            gold_unclear_total += 1
            confusion[f"{row.get('rule')}->{row.get('model')}"] += 1
        else:
            gold_absent_total += 1
            confusion[f"{row.get('rule')}->{row.get('model')}"] += 1
        if confidence is not None and matched_value in {"Да", "Нет"}:
            if matched:
                confidence_correct.append(confidence)
            else:
                confidence_errors.append(confidence)
                if confidence >= 0.8 and row.get("error_type") not in {"both_correct", "no_gold_agree"}:
                    high_conf_wrong.append(dict(row))
    return {
        "rows_total": len(rows),
        "model_fix": int(type_counts.get("model_fix", 0) + type_counts.get("true_fix", 0)),
        "model_break": int(type_counts.get("model_break", 0) + type_counts.get("false_fix", 0) + type_counts.get("false_negative", 0)),
        "both_wrong": int(type_counts.get("both_wrong", 0)),
        "error_type_counts": dict(type_counts.most_common()),
        "guard_effect_type_counts": dict(guard_effect_counts.most_common()),
        "avg_confidence_correct": average(confidence_correct),
        "avg_confidence_errors": average(confidence_errors),
        "high_conf_wrong_count": len(high_conf_wrong),
        "gold_present_total": gold_present_total,
        "gold_absent_total": gold_absent_total,
        "gold_unclear_total": gold_unclear_total,
        "excluded_from_confusion_total": 0,
        "high_conf_wrong": [
            {key: row.get(key, "") for key in ("id", "gold", "rule", "model", "confidence", "error_type", "rationale")}
            for row in high_conf_wrong[:50]
        ],
        "confusion_matrix": dict(confusion.most_common()),
    }


def classify_error_type(*, gold: str, rule: str, model: str) -> str:
    if not gold:
        return "no_gold_agree" if rule == model else "no_gold_rule_model_disagree"
    rule_ok = bool(rule and rule == gold)
    model_ok = bool(model and model == gold)
    if rule_ok and model_ok:
        return "both_correct"
    if model_ok and not rule_ok:
        return "model_fix"
    if rule_ok and not model_ok:
        return "model_break"
    return "both_wrong"


def guard_effect_type(
    *,
    gold: str,
    raw_model: str,
    post_model: str,
    is_guarded: bool,
) -> str:
    if not is_guarded:
        return "no_guard"
    if not gold:
        return "guarded_no_gold"
    raw_ok = bool(raw_model and raw_model == gold)
    post_ok = bool(post_model and post_model == gold)
    if not raw_ok and post_ok:
        return "fixed"
    if raw_ok and not post_ok:
        return "broke"
    if raw_ok and post_ok:
        return "neutral_correct"
    return "neutral_wrong"


def review_gold_label(verdict: str, *, rule: str, model: str, block: str) -> str:
    if verdict == "true_fix":
        return model
    if verdict == "false_fix":
        return rule
    if block == "e" and verdict == "expected_fail_closed":
        return model
    if block == "e" and verdict == "false_negative":
        return rule
    return ""


def review_error_type(verdict: str, *, block: str, gold: str, rule: str, model: str) -> str:
    if verdict == "unclear":
        return "gold_unclear"
    if gold:
        return classify_error_type(gold=gold, rule=rule, model=model)
    return f"{block}_review_unknown"


def split_flip(value: str) -> tuple[str, str]:
    if "->" not in value:
        return value, ""
    left, right = value.split("->", 1)
    return left.strip(), right.strip()


def parse_blocks(value: Any) -> list[str]:
    raw = str(value or "all").strip().lower()
    if raw in {"", "all", "*"}:
        return ["c", "d", "a", "b", "e"]
    blocks: list[str] = []
    for part in raw.replace(";", ",").split(","):
        block = part.strip().lower()
        if not block:
            continue
        if block not in {"a", "b", "c", "d", "e"}:
            raise SystemExit(f"Unsupported block for TZ-117 trace: {block}")
        if block not in blocks:
            blocks.append(block)
    if not blocks:
        raise SystemExit("At least one TZ-117 trace block is required")
    return blocks


def combine_verdict(value: Mapping[str, Any]) -> str:
    verdict = str(value.get("close_verdict") or "")
    risk = str(value.get("premature_close_risk") or "")
    return f"{verdict}|{risk}".strip("|")


def redact_fragment(value: Any, *, limit: int = 240) -> str:
    text = " ".join(str(value or "").split())
    text = re.sub(r"[\w.+-]+@[\w.-]+\.[A-Za-zА-Яа-я]{2,}", "[email]", text)
    text = re.sub(r"(?:\+?\d[\d\s().-]{8,}\d)", "[phone]", text)
    if len(text) > limit:
        text = text[: max(0, limit - 1)].rstrip() + "…"
    return text


def one_line(value: Any, *, limit: int = 260) -> str:
    return redact_fragment(value, limit=limit)


def bool_cell(value: bool) -> str:
    return "Да" if value else "Нет"


def parse_float(value: Any) -> float | None:
    try:
        text = str(value or "").strip().replace(",", ".")
        return float(text) if text else None
    except ValueError:
        return None


def average(values: Sequence[float]) -> float | None:
    return round(sum(values) / len(values), 6) if values else None


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            if isinstance(payload, dict):
                yield payload


def write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    fields = TRACE_FIELDS if rows else TRACE_FIELDS
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def render_trace_report(block: str, summary: Mapping[str, Any], rows: list[dict[str, str]]) -> str:
    error_rows = [
        row
        for row in rows
        if row.get("error_type") not in {"both_correct", "model_fix", "true_fix", "expected_fail_closed", "no_gold_agree"}
    ][:15]
    lines = [
        f"# TZ-117 {block.upper()} Trace Report",
        "",
        f"- Rows: `{summary['rows_total']}`",
        f"- model_fix: `{summary['model_fix']}`",
        f"- model_break: `{summary['model_break']}`",
        f"- both_wrong: `{summary['both_wrong']}`",
        f"- avg_confidence_correct: `{summary['avg_confidence_correct']}`",
        f"- avg_confidence_errors: `{summary['avg_confidence_errors']}`",
        f"- high_conf_wrong_count: `{summary['high_conf_wrong_count']}`",
        f"- gold_present_total: `{summary['gold_present_total']}`",
        f"- gold_absent_total: `{summary['gold_absent_total']}`",
        f"- gold_unclear_total: `{summary['gold_unclear_total']}`",
        "",
        "## Error Examples",
        "",
        "| id | gold | rule | model | conf | type | rationale |",
        "|---|---|---|---|---:|---|---|",
    ]
    for row in error_rows:
        lines.append(
            f"| `{row['id']}` | `{row['gold']}` | `{row['rule']}` | `{row['model']}` | "
            f"`{row['confidence']}` | `{row['error_type']}` | {row['rationale']} |"
        )
    return "\n".join(lines) + "\n"


def render_combined_report(combined: Mapping[str, Any]) -> str:
    lines = [
        "# TZ-117 Error Trace Combined Report",
        "",
        f"- Generated: `{combined['generated_at']}`",
        f"- Out dir: `{combined['out_dir']}`",
        "",
        "| block | rows | model_fix | model_break | both_wrong | high_conf_wrong |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for block, summary in combined["blocks"].items():
        lines.append(
            f"| `{block}` | `{summary['rows_total']}` | `{summary['model_fix']}` | "
            f"`{summary['model_break']}` | `{summary['both_wrong']}` | `{summary['high_conf_wrong_count']}` |"
        )
    lines.extend(
        [
            "",
            "Safety: trace only, no new model calls, no primary/writeback, no CRM/Tallanto/stable_runtime writes.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TZ-117: build unified error traces for A/B/C/D/E offline blocks.")
    parser.add_argument("--out-dir", default="audits/_inbox/tz117_error_traces")
    parser.add_argument("--c-dir", default=str(DEFAULT_C))
    parser.add_argument("--a-dir", default=str(DEFAULT_A))
    parser.add_argument("--d-dir", default=str(DEFAULT_D))
    parser.add_argument("--gold-dir", default=str(DEFAULT_GOLD))
    parser.add_argument("--blocks", default="all", help="Comma-separated blocks to build: a,b,c,d,e or all.")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
