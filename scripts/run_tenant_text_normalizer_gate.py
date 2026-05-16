#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mango_mvp.quality.tenant_text_normalizer import (
    detect_product_list_artifacts,
    detect_residual_manager_text_artifacts,
    format_product_list,
    normalize_manager_text,
)


PRODUCT_FIELD_MARKERS = ("продукт", "интерес")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gate tenant text normalizer coverage on CSV exports.")
    parser.add_argument("--input", action="append", required=True, help="CSV file to scan. Can be passed multiple times.")
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--fail-on-residual", action="store_true", default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_root = Path(args.out_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    report_rows: list[dict[str, Any]] = []
    normalization_rows: list[dict[str, Any]] = []
    class_counts: Counter[str] = Counter()
    pre_class_counts: Counter[str] = Counter()
    rows_scanned = 0
    cells_scanned = 0
    changed_cells = 0
    material_changed_cells = 0

    for input_value in args.input:
        path = Path(input_value).expanduser().resolve()
        for row_index, row in enumerate(_read_csv(path), start=1):
            rows_scanned += 1
            for field, raw_value in row.items():
                value = _safe_text(raw_value)
                if not value:
                    continue
                cells_scanned += 1
                normalized = normalize_manager_text(value)
                if normalized != value:
                    changed_cells += 1
                pre_findings = detect_residual_manager_text_artifacts(value)
                findings = detect_residual_manager_text_artifacts(normalized)
                product_normalized = ""
                if _is_product_field(field):
                    product_normalized = format_product_list(value)
                    if product_normalized and product_normalized != value:
                        changed_cells += 1
                    pre_findings.extend(detect_product_list_artifacts(value))
                    findings.extend(detect_product_list_artifacts(product_normalized or normalized))
                if pre_findings:
                    material_changed_cells += 1
                    for finding in pre_findings:
                        pre_class_counts[finding.class_id] += 1
                    normalization_rows.append(
                        {
                            "input": str(path),
                            "row_index": row_index,
                            "field": field,
                            "class_ids": " | ".join(sorted({item.class_id for item in pre_findings})),
                            "raw_preview": value[:600],
                            "normalized_preview": (product_normalized or normalized)[:600],
                        }
                    )
                for finding in findings:
                    class_counts[finding.class_id] += 1
                    report_rows.append(
                        {
                            "input": str(path),
                            "row_index": row_index,
                            "field": field,
                            "class_id": finding.class_id,
                            "matched_text": finding.matched_text,
                            "reason": finding.reason,
                            "value_preview": value[:600],
                            "normalized_preview": normalized[:600],
                        }
                    )

    outputs = {
        "report_csv": out_root / "tenant_text_normalizer_gate_report.csv",
        "normalization_changes_csv": out_root / "tenant_text_normalizer_normalization_changes.csv",
        "summary_json": out_root / "summary.json",
    }
    _write_csv(outputs["report_csv"], report_rows)
    _write_normalization_csv(outputs["normalization_changes_csv"], normalization_rows)
    passed = not report_rows
    summary = {
        "schema_version": "tenant_text_normalizer_gate_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "inputs": [str(Path(item).expanduser().resolve()) for item in args.input],
        "passed": passed,
        "rows_scanned": rows_scanned,
        "cells_scanned": cells_scanned,
        "changed_cells": changed_cells,
        "material_changed_cells": material_changed_cells,
        "pre_normalization_findings": len(normalization_rows),
        "pre_class_counts": dict(pre_class_counts),
        "residual_findings": len(report_rows),
        "class_counts": dict(class_counts),
        "outputs": {key: str(path) for key, path in outputs.items()},
    }
    outputs["summary_json"].write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if args.fail_on_residual and not passed else 0


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "input",
        "row_index",
        "field",
        "class_id",
        "matched_text",
        "reason",
        "value_preview",
        "normalized_preview",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_normalization_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "input",
        "row_index",
        "field",
        "class_ids",
        "raw_preview",
        "normalized_preview",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _safe_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _is_product_field(field: str) -> bool:
    lower = field.casefold()
    return any(marker in lower for marker in PRODUCT_FIELD_MARKERS)


if __name__ == "__main__":
    raise SystemExit(main())
