#!/usr/bin/env python3
"""Build aggregate triage reports for the canonical read-only customer timeline."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mango_mvp.customer_timeline.canonical_readonly_import import DEFAULT_OUT_ROOT  # noqa: E402
from mango_mvp.customer_timeline.canonical_readonly_triage import (  # noqa: E402
    CanonicalReadonlyTriageConfig,
    build_canonical_readonly_timeline_triage,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build read-only aggregate triage reports for customer_timeline.")
    parser.add_argument("--project-root", default=str(ROOT), help="Project root. Defaults to this repository.")
    parser.add_argument("--timeline-root", default=str(DEFAULT_OUT_ROOT), help="Existing canonical timeline root.")
    parser.add_argument("--out-dir", help="Output directory under timeline root. Defaults to <timeline-root>/triage.")
    parser.add_argument("--generated-at", help="UTC ISO timestamp for deterministic tests/rebuilds.")
    return parser.parse_args(argv)


def parse_generated_at(value: str | None) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_canonical_readonly_timeline_triage(
        CanonicalReadonlyTriageConfig(
            project_root=Path(args.project_root).expanduser(),
            timeline_root=Path(args.timeline_root).expanduser(),
            out_dir=Path(args.out_dir).expanduser() if args.out_dir else None,
            generated_at=parse_generated_at(args.generated_at),
        )
    )
    print(json.dumps({"summary": report["summary"], "paths": report.get("paths", {})}, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
