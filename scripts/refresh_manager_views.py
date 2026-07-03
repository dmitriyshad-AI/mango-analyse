#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from mango_mvp.customer_timeline.safety import guard_customer_timeline_output_path  # noqa: E402
from build_manager_dossier import build_manager_dossier_workbook  # noqa: E402
from build_wave0_manager_lists import build_workbook as build_wave0_workbook  # noqa: E402


DEFAULT_TIMELINE_DB = Path(".codex_local/staging/customer_timeline_staging.sqlite")
DEFAULT_RECONCILE_JSON = Path(".codex_local/review/f9_amo_actuality/stalled_deals_amo_reconcile.json")


def refresh_manager_views(
    *,
    timeline_db: Path,
    allowed_root: Path,
    out_dir: Path,
    reconcile_json: Path | None,
    customer_ids_file: Path | None,
    canonical_calls_db: Path | None,
    limit: int,
    run: bool,
) -> dict:
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if not run:
        return {
            "status": "skipped_off_by_default",
            "generated_at": generated_at,
            "reason": "pass --run to build local manager views",
            "writes": [],
        }
    out_root = _guard_local_manager_views_dir(out_dir, allowed_root)
    stamp = generated_at.replace(":", "").replace("+", "Z")
    out_root.mkdir(parents=True, exist_ok=True)
    customer_ids = _read_customer_ids(customer_ids_file)
    wave0 = build_wave0_workbook(
        timeline_db=timeline_db,
        out_xlsx=out_root / f"{stamp}_wave0_dengi_na_polu.xlsx",
        allowed_root=allowed_root,
        reconcile_json=reconcile_json,
        limit_per_sheet=500,
    )
    dossier = build_manager_dossier_workbook(
        timeline_db=timeline_db,
        allowed_root=allowed_root,
        out_xlsx=out_root / f"{stamp}_wave1_manager_dossier.xlsx",
        customer_ids=tuple(customer_ids),
        canonical_calls_db=canonical_calls_db,
        reconcile_json=reconcile_json,
        limit=limit,
    )
    return {
        "status": "built",
        "generated_at": generated_at,
        "wave0": wave0,
        "dossier": dossier,
    }


def _read_customer_ids(path: Path | None) -> list[str]:
    if path is None:
        return []
    return [line.strip() for line in path.expanduser().read_text(encoding="utf-8").splitlines() if line.strip() and not line.lstrip().startswith("#")]


def _guard_local_manager_views_dir(path: Path, allowed_root: Path) -> Path:
    resolved = guard_customer_timeline_output_path(path, allowed_root)
    root = Path(allowed_root).expanduser().resolve(strict=False)
    relative = resolved.relative_to(root)
    if not relative.parts or relative.parts[0] != ".codex_local":
        raise ValueError("manager views contain PII and must stay under .codex_local")
    return resolved


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh local manager views. OFF by default; pass --run to write under .codex_local.")
    parser.add_argument("--run", action="store_true", help="Actually build local manager workbooks.")
    parser.add_argument("--timeline-db", type=Path, default=DEFAULT_TIMELINE_DB)
    parser.add_argument("--allowed-root", type=Path, default=ROOT)
    parser.add_argument("--out-dir", type=Path, default=Path(".codex_local/review/manager_views"))
    parser.add_argument("--reconcile-json", type=Path, default=DEFAULT_RECONCILE_JSON)
    parser.add_argument("--customer-ids-file", type=Path)
    parser.add_argument("--canonical-calls-db", type=Path)
    parser.add_argument("--limit", type=int, default=50)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = refresh_manager_views(
        timeline_db=args.timeline_db,
        allowed_root=args.allowed_root,
        out_dir=args.out_dir,
        reconcile_json=args.reconcile_json,
        customer_ids_file=args.customer_ids_file,
        canonical_calls_db=args.canonical_calls_db,
        limit=args.limit,
        run=args.run,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
