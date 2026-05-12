from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mango_mvp.quality.crm_writeback_quality_detector import (
    detect_crm_writeback_quality_risks,
    findings_to_risk_counts,
)


@dataclass(frozen=True)
class CrmWritebackCorpusValidationConfig:
    corpus_jsonl: Path
    out_root: Path
    detector_min_severity: str = "P2"


def validate_crm_writeback_frozen_corpus(config: CrmWritebackCorpusValidationConfig) -> dict[str, Any]:
    out_root = config.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    cases = _read_jsonl(config.corpus_jsonl)

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for case in cases:
        validation = validate_one_case(case, detector_min_severity=config.detector_min_severity)
        rows.append(validation)
        if validation["passed"] != "yes":
            failures.append(validation)

    outputs = {
        "validation_results_csv": out_root / "validation_results.csv",
        "validation_failures_csv": out_root / "validation_failures.csv",
        "summary_json": out_root / "summary.json",
        "report_md": out_root / "CRM_WRITEBACK_FROZEN_CORPUS_VALIDATION.md",
    }
    _write_csv(outputs["validation_results_csv"], rows)
    _write_csv(outputs["validation_failures_csv"], failures, fieldnames=list(rows[0].keys()) if rows else [])
    seed_policy = _seed_policy_summary(rows)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "corpus_jsonl": str(config.corpus_jsonl.resolve()),
        "detector_min_severity": config.detector_min_severity,
        "rows": len(rows),
        "passed": len(failures) == 0,
        "failures": len(failures),
        "by_expected_decision": dict(Counter(row["expected_decision"] for row in rows).most_common()),
        "by_layer": dict(Counter(row["layer"] for row in rows).most_common()),
        "seed_policy": seed_policy,
        "rolling_closure": {
            "class_id": "C1/F5+C8/F8",
            "status": "monitoring",
            "can_claim_closed": False,
            "reasons": [
                "Requires population recall gate and at least one follow-up run before closure",
                *([] if seed_policy["external_seed_ratio"] >= 0.5 else ["external_seed_ratio < 0.5"]),
            ],
        },
        "risk_counts": dict(Counter(risk for row in rows for risk in _split(row["detector_risk_types"])).most_common()),
        "outputs": {key: str(path) for key, path in outputs.items()},
    }
    outputs["summary_json"].write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    outputs["report_md"].write_text(_validation_report(summary), encoding="utf-8")
    return summary


def validate_one_case(case: dict[str, Any], *, detector_min_severity: str = "P2") -> dict[str, Any]:
    text = str(case.get("input_text") or "")
    expected = str(case.get("expected_decision") or "").strip().casefold()
    findings = detect_crm_writeback_quality_risks(text, min_severity=detector_min_severity)
    actual = "block" if findings else "allow"
    passed = actual == expected
    risk_counts = findings_to_risk_counts(findings)
    return {
        "case_id": case.get("case_id", ""),
        "layer": case.get("layer", ""),
        "seed_source": case.get("seed_source", ""),
        "closure_class_id": case.get("closure_class_id", ""),
        "risk_class": case.get("risk_class", ""),
        "expected_decision": expected,
        "actual_decision": actual,
        "passed": "yes" if passed else "no",
        "detector_findings": len(findings),
        "detector_risk_types": " | ".join(sorted(risk_counts)),
        "detector_matches": " | ".join(f"{finding.risk_type}:{finding.matched_text}" for finding in findings[:10]),
        "input_text": text,
    }


def _seed_policy_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    block_rows = [row for row in rows if row.get("expected_decision") == "block"]
    prior_audit_rows = [row for row in block_rows if str(row.get("layer", "")).startswith("claude_v")]
    external_rows = [row for row in block_rows if row not in prior_audit_rows]
    external_ratio = round(len(external_rows) / len(block_rows), 4) if block_rows else 1.0
    by_seed_source = dict(Counter(row.get("seed_source", "") or "<missing>" for row in rows).most_common())
    return {
        "schema_version": "crm_writeback_corpus_seed_policy_v1",
        "block_rows": len(block_rows),
        "prior_audit_block_rows": len(prior_audit_rows),
        "external_block_rows": len(external_rows),
        "external_seed_ratio": external_ratio,
        "passed_minimum_external_seed_ratio": external_ratio >= 0.5,
        "by_seed_source": by_seed_source,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            raw = line.strip()
            if raw:
                rows.append(json.loads(raw))
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _split(value: str) -> list[str]:
    return [part.strip() for part in str(value or "").split("|") if part.strip()]


def _validation_report(summary: dict[str, Any]) -> str:
    verdict = "PASS" if summary.get("passed") else "FAIL"
    return "\n".join(
        [
            "# CRM Writeback Frozen Corpus Validation",
            "",
            f"Verdict: `{verdict}`",
            f"Rows: `{summary.get('rows')}`",
            f"Failures: `{summary.get('failures')}`",
            f"Detector min severity: `{summary.get('detector_min_severity')}`",
            "",
            "## Outputs",
            *[f"- `{key}`: `{value}`" for key, value in (summary.get("outputs") or {}).items()],
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate CRM writeback EdTech relevance frozen corpus.")
    parser.add_argument("--corpus-jsonl", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--detector-min-severity", default="P2")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = validate_crm_writeback_frozen_corpus(
        CrmWritebackCorpusValidationConfig(
            corpus_jsonl=Path(args.corpus_jsonl),
            out_root=Path(args.out_root),
            detector_min_severity=args.detector_min_severity,
        )
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
