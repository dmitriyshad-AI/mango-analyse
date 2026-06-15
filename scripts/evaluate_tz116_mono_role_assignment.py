#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


MODES = {"off", "shadow", "primary"}
VALID_ROLES = {"manager", "client"}


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    mode = normalize_mode(args.mode)
    rows = list(read_jsonl(Path(args.input).expanduser().resolve()))
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    result_rows = [evaluate_case(row, mode=mode) for row in rows]
    counters = Counter(str(row["status"]) for row in result_rows)
    provider_counts = Counter(str(row.get("provider") or "") for row in result_rows)
    comparable = [row for row in result_rows if row["status"] == "compared"]
    correct = sum(1 for row in comparable if row["exact_match"] == "1")
    summary = {
        "schema_version": "tz116_mono_role_assignment_eval_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "input": str(Path(args.input).expanduser().resolve()),
        "total": len(rows),
        "comparable": len(comparable),
        "exact_match": correct,
        "exact_match_rate": round(correct / len(comparable), 6) if comparable else 0.0,
        "status_counts": dict(counters),
        "provider_counts": dict(provider_counts),
        "llm_calls_total": 0,
        "safety": {
            "calls_openai": False,
            "reads_audio": False,
            "runs_asr": False,
            "uses_real_call_text": False,
        },
    }
    write_csv(out_dir / "mono_role_assignment_eval_rows.csv", result_rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "REPORT.md").write_text(render_report(summary, result_rows), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def evaluate_case(row: dict[str, Any], *, mode: str) -> dict[str, str]:
    case_id = str(row.get("case_id") or row.get("id") or "")
    gold = normalize_roles(row.get("gold_roles"))
    rule = normalize_roles(row.get("rule_roles"))
    model = normalize_roles(row.get("model_roles"))
    if mode == "off":
        predicted: list[str] = []
        provider = "off"
    elif mode == "shadow":
        predicted = rule
        provider = "rule_shadow"
    else:
        predicted = model
        provider = "precomputed_model_primary"

    status = "compared" if gold and len(gold) == len(predicted) else "not_comparable"
    exact = "1" if status == "compared" and gold == predicted else "0"
    shadow_disagreement = "1" if mode == "shadow" and rule and model and rule != model else "0"
    return {
        "case_id": case_id,
        "mode": mode,
        "provider": provider,
        "turn_count": str(max(len(gold), len(rule), len(model), len(row.get("turns") or []))),
        "status": status,
        "exact_match": exact,
        "shadow_rule_model_disagreement": shadow_disagreement,
        "gold_roles": " ".join(gold),
        "rule_roles": " ".join(rule),
        "model_roles": " ".join(model),
        "predicted_roles": " ".join(predicted),
    }


def normalize_roles(value: Any) -> list[str]:
    if isinstance(value, str):
        raw = [part for part in value.replace(",", " ").split() if part]
    elif isinstance(value, list):
        raw = [str(item) for item in value]
    else:
        raw = []
    return [role for role in (item.strip().lower() for item in raw) if role in VALID_ROLES]


def normalize_mode(value: Any) -> str:
    mode = str(value or "off").strip().lower()
    return mode if mode in MODES else "off"


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                yield parsed


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fields = list(rows[0]) if rows else ["empty"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def render_report(summary: dict[str, Any], rows: list[dict[str, str]]) -> str:
    lines = [
        "# TZ-116 D Mono Role Assignment Evaluation",
        "",
        f"- Mode: `{summary['mode']}`",
        f"- Total: `{summary['total']}`",
        f"- Comparable: `{summary['comparable']}`",
        f"- Exact match rate: `{summary['exact_match_rate']}`",
        f"- LLM calls total: `{summary['llm_calls_total']}`",
        "",
        "Safety: synthetic/gold rows only; no OpenAI, no audio, no ASR.",
        "",
        "## First Rows",
        "",
        "| case_id | status | exact | provider | disagreement |",
        "|---|---|---:|---|---:|",
    ]
    for row in rows[:30]:
        lines.append(
            f"| `{row['case_id']}` | `{row['status']}` | `{row['exact_match']}` | "
            f"`{row['provider']}` | `{row['shadow_rule_model_disagreement']}` |"
        )
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TZ-116 D: evaluate synthetic/gold mono role assignment rows without OpenAI or ASR.")
    parser.add_argument("--input", required=True, help="JSONL with case_id, gold_roles and optional rule_roles/model_roles.")
    parser.add_argument("--out-dir", default="audits/_inbox/tz116_mono_role_assignment_eval")
    parser.add_argument("--mode", choices=sorted(MODES), default="off")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
