#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from mango_mvp.amocrm_runtime import deals as deals_module


MODES = {"off", "shadow", "primary"}


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    mode = normalize_mode(args.mode)
    records = list(read_jsonl(Path(args.input)))
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    for index, record in enumerate(records, start=1):
        case_id = str(record.get("case_id") or record.get("id") or index)
        heuristic = dict(record.get("heuristic_analysis") or {})
        llm = record.get("llm_analysis")
        if not heuristic:
            counters["missing_heuristic"] += 1
            continue
        if mode in {"shadow", "primary"} and not isinstance(llm, dict):
            counters["missing_llm_analysis"] += 1
        final, comparison = finalize_offline(heuristic, llm if isinstance(llm, dict) else None, mode=mode)
        row = build_row(case_id, heuristic, llm if isinstance(llm, dict) else None, final, comparison)
        rows.append(row)
        results.append(
            {
                "case_id": case_id,
                "mode": mode,
                "heuristic_analysis": heuristic,
                "llm_analysis": llm if isinstance(llm, dict) else None,
                "final_analysis": final,
                "comparison": comparison,
            }
        )
        counters["processed"] += 1
        if comparison:
            counters["compared"] += 1
            if comparison.get("verdict_changed"):
                counters["verdict_changed"] += 1
            if comparison.get("risk_changed"):
                counters["risk_changed"] += 1
            if comparison.get("severe_conflict"):
                counters["severe_conflict"] += 1
        if final.get("writeback_allowed"):
            counters["would_allow_writeback"] += 1

    disagreements = [row for row in rows if row.get("verdict_changed") == "Да" or row.get("risk_changed") == "Да"]
    summary = {
        "schema_version": "tz116_crm_llm_offline_measure_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "input": str(Path(args.input).expanduser().resolve()),
        "records_total": len(records),
        "counters": dict(counters),
        "llm_calls_total": 0,
        "safety": {
            "calls_live_llm": False,
            "writes_amo": False,
            "writes_tallanto": False,
            "writes_crm": False,
            "uses_cache": False,
        },
    }
    write_csv(out_dir / "crm_llm_offline_measure_rows.csv", rows)
    write_csv(out_dir / "crm_llm_offline_measure_disagreements.csv", disagreements)
    write_jsonl(out_dir / "crm_llm_offline_measure_results.jsonl", results)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "REPORT.md").write_text(render_report(summary, disagreements), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def normalize_mode(value: Any) -> str:
    mode = str(value or "off").strip().lower()
    return mode if mode in MODES else "off"


def finalize_offline(
    heuristic: dict[str, Any],
    llm: dict[str, Any] | None,
    *,
    mode: str,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    resolved_mode = normalize_mode(mode)
    if resolved_mode == "off" or llm is None:
        final = dict(heuristic)
        final.setdefault("analysis_source", "heuristic")
        final.setdefault("analysis_mode", "heuristic")
        final.setdefault("writeback_allowed", False)
        final.setdefault("writeback_blockers", ["offline_measure_no_writeback"])
        return final, None

    crm_mode = "llm_shadow" if resolved_mode == "shadow" else "llm_primary"
    comparison = deals_module._comparison_summary(heuristic, llm)  # noqa: SLF001
    final = {
        **heuristic,
        **llm,
        "analysis_source": "llm",
        "analysis_mode": crm_mode,
    }
    blockers = deals_module._writeback_blockers(analysis=final, mode=crm_mode, comparison=comparison)  # noqa: SLF001
    if "offline_measure_no_writeback" not in blockers:
        blockers.append("offline_measure_no_writeback")
    final["writeback_allowed"] = False
    final["writeback_blockers"] = blockers
    final["heuristic_llm_comparison"] = comparison
    return final, comparison


def build_row(
    case_id: str,
    heuristic: dict[str, Any],
    llm: dict[str, Any] | None,
    final: dict[str, Any],
    comparison: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "heuristic_verdict": heuristic.get("close_verdict", ""),
        "llm_verdict": (llm or {}).get("close_verdict", ""),
        "final_verdict": final.get("close_verdict", ""),
        "heuristic_risk": heuristic.get("premature_close_risk", ""),
        "llm_risk": (llm or {}).get("premature_close_risk", ""),
        "final_risk": final.get("premature_close_risk", ""),
        "verdict_changed": "Да" if comparison and comparison.get("verdict_changed") else "Нет",
        "risk_changed": "Да" if comparison and comparison.get("risk_changed") else "Нет",
        "severe_conflict": "Да" if comparison and comparison.get("severe_conflict") else "Нет",
        "final_writeback_allowed": "Да" if final.get("writeback_allowed") else "Нет",
        "final_writeback_blockers": " | ".join(str(item) for item in final.get("writeback_blockers", [])),
    }


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:  # type: ignore[arg-type]
        for line in handle:
            text = line.strip()
            if not text:
                continue
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                yield parsed


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render_report(summary: dict[str, Any], disagreements: list[dict[str, Any]]) -> str:
    counters = summary["counters"]
    lines = [
        "# TZ-116 A CRM LLM Offline Measurement",
        "",
        f"- Mode: `{summary['mode']}`",
        f"- Records: `{summary['records_total']}`",
        f"- Compared: `{counters.get('compared', 0)}`",
        f"- Verdict changed: `{counters.get('verdict_changed', 0)}`",
        f"- Severe conflict: `{counters.get('severe_conflict', 0)}`",
        f"- LLM calls total: `{summary['llm_calls_total']}`",
        "",
        "Safety: no live LLM calls, no cache, no writeback.",
        "",
        "## First Disagreements",
        "",
        "| case_id | heuristic | llm | severe | blockers |",
        "|---|---|---|---|---|",
    ]
    for row in disagreements[:30]:
        lines.append(
            f"| `{row['case_id']}` | `{row['heuristic_verdict']}` | `{row['llm_verdict']}` | "
            f"`{row['severe_conflict']}` | {str(row['final_writeback_blockers']).replace('|', '/')} |"
        )
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TZ-116 A: compare precomputed CRM LLM analysis with heuristic offline.")
    parser.add_argument("--input", required=True, help="JSONL/JSONL.GZ with case_id, heuristic_analysis, optional llm_analysis.")
    parser.add_argument("--out-dir", default="audits/_inbox/tz116_crm_llm_offline_measure")
    parser.add_argument("--mode", choices=sorted(MODES), default="off")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
