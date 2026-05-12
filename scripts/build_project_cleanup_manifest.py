#!/usr/bin/env python3
"""Build a read-only project cleanup manifest."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mango_mvp.productization.project_cleanup_manifest import (  # noqa: E402
    DEFAULT_PROJECT_CLEANUP_MANIFEST_ROOT,
    ProjectCleanupManifestConfig,
    build_project_cleanup_manifest,
)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        summary = build_project_cleanup_manifest(
            ProjectCleanupManifestConfig(
                project_root=Path(args.project_root),
                out_root=Path(args.out_root),
                current_runtime_path=Path(args.current_runtime) if args.current_runtime else None,
                generated_at=_parse_datetime(args.generated_at),
                fresh_audit_days=args.fresh_audit_days,
            )
        )
    except Exception as exc:
        print(f"project cleanup manifest failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"summary": summary, "safety": summary["safety"]}, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a fail-safe cleanup manifest. This command never deletes, moves or quarantines files."
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--out-root", default=str(DEFAULT_PROJECT_CLEANUP_MANIFEST_ROOT))
    parser.add_argument("--current-runtime", default=str(Path("stable_runtime") / "CURRENT_RUNTIME.json"))
    parser.add_argument(
        "--generated-at",
        default="",
        help="Optional ISO timestamp for deterministic fresh-audit cutoff calculations.",
    )
    parser.add_argument(
        "--fresh-audit-days",
        type=int,
        default=1,
        help="Audit packs with a date token inside this lookback window are excluded from candidates.",
    )
    return parser.parse_args(argv)


def _parse_datetime(value: str) -> datetime | None:
    text = (value or "").strip()
    if not text:
        return None
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


if __name__ == "__main__":
    raise SystemExit(main())
