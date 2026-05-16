#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from mango_mvp.question_catalog.rebuild_from_predictions import rebuild_catalog_from_predictions


DEFAULT_ITEMS = Path("product_data/question_catalog/customer_question_items.jsonl")


def main() -> None:
    args = parse_args()
    since = datetime.fromisoformat(str(args.since).replace("Z", "+00:00"))
    if since.tzinfo is None or since.utcoffset() is None:
        since = since.replace(tzinfo=timezone.utc)
    summary = rebuild_catalog_from_predictions(
        project_root=Path(args.project_root).resolve(),
        items_path=Path(args.items).resolve(),
        predictions_path=Path(args.predictions).resolve(),
        out_root=Path(args.out_root).resolve(),
        tenant_id=args.tenant_id,
        since=since,
        require_all_predictions=not args.allow_partial,
    )
    print(
        json.dumps(
            {
                "out_root": str(Path(args.out_root).resolve()),
                "question_items": summary["totals"]["question_items"],
                "question_classes": summary["totals"]["question_classes"],
                "dynamic_fact_classes": summary["totals"]["dynamic_fact_classes"],
                "summary": str(Path(args.out_root).resolve() / "question_catalog_summary.json"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild question catalog from Codex full-run predictions.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--items", default=str(DEFAULT_ITEMS))
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--tenant-id", default="foton")
    parser.add_argument("--since", default="2025-01-01T00:00:00+00:00")
    parser.add_argument("--allow-partial", action="store_true", help="Keep original item classification when a prediction is missing.")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
