#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mango_mvp.question_catalog.rop_policy_import import (
    DEFAULT_APPROVED_ROP_WORKBOOK,
    DEFAULT_TAXONOMY_PATH,
    DEFAULT_V2_CSV,
    DEFAULT_V2_XLSX,
    apply_rop_policies_to_taxonomy,
    build_final_v2_rows,
    load_approved_rop_policies,
    load_taxonomy,
    write_final_v2_csv,
    write_final_v2_xlsx,
    write_taxonomy,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply approved ROP policy workbook to Question Catalog v2 taxonomy.")
    parser.add_argument("--workbook", default=str(DEFAULT_APPROVED_ROP_WORKBOOK))
    parser.add_argument("--taxonomy", default=str(DEFAULT_TAXONOMY_PATH))
    parser.add_argument("--out-csv", default=str(DEFAULT_V2_CSV))
    parser.add_argument("--out-xlsx", default=str(DEFAULT_V2_XLSX))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    workbook_path = Path(args.workbook)
    taxonomy_path = Path(args.taxonomy)
    policies = load_approved_rop_policies(workbook_path)
    taxonomy = load_taxonomy(taxonomy_path)
    updated = apply_rop_policies_to_taxonomy(taxonomy, policies)
    rows = build_final_v2_rows(updated)
    if not args.dry_run:
        write_taxonomy(taxonomy_path, updated)
        write_final_v2_csv(args.out_csv, rows)
        write_final_v2_xlsx(args.out_xlsx, rows)
    decisions = {}
    for row in rows:
        decisions[row["Решение РОПа"]] = decisions.get(row["Решение РОПа"], 0) + 1
    print(
        json.dumps(
            {
                "status": "dry_run" if args.dry_run else "ok",
                "workbook": str(workbook_path),
                "taxonomy": str(taxonomy_path),
                "themes": len(rows),
                "decisions": decisions,
                "out_csv": str(args.out_csv),
                "out_xlsx": str(args.out_xlsx),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
