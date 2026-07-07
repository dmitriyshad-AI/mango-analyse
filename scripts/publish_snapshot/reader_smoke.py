#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mango_mvp.customer_timeline.read_api import CustomerTimelineReadApi, CustomerTimelineReadApiConfig  # noqa: E402
from scripts.publish_snapshot.common import (  # noqa: E402
    add_common_args,
    finish_cli,
    load_config,
    quick_check,
    render_command,
    report_base,
    run_command,
)


def run_internal_smoke(db_path: Path, allowed_root: Path, tenant_id: str, control_customers: tuple[dict, ...]) -> list[dict]:
    results = []
    with CustomerTimelineReadApi.open(CustomerTimelineReadApiConfig(timeline_db=db_path, allowed_root=allowed_root)) as api:
        for item in control_customers:
            customer_id = str(item.get("customer_id") or "")
            profile = api.customer_profile(tenant_id, customer_id, event_limit=5, bot_context_limit=5)
            results.append(
                {
                    "customer_id": customer_id,
                    "expected_found": bool(item.get("expected_found", True)),
                    "found": bool(profile.get("found")),
                    "events": len(((profile.get("timeline") or {}).get("items") or [])),
                    "bot_visible": int(((profile.get("bot_context") or {}).get("summary") or {}).get("visible_chunks") or 0),
                    "ok": bool(profile.get("found")) == bool(item.get("expected_found", True)),
                }
            )
    return results


def smoke(config_path: Path, *, snapshot_db: Path) -> tuple[dict, bool]:
    cfg = load_config(config_path)
    report = report_base(cfg, "reader_smoke")
    db_path = snapshot_db.expanduser().resolve(strict=False)
    variables = {"db": db_path, "allowed_root": db_path.parent, "tenant_id": cfg.tenant_id}
    reader_results = []
    ok = quick_check(db_path) == "ok"
    for reader in cfg.readers:
        command = reader.get("smoke_command")
        if command:
            worktree = Path(str(reader.get("worktree") or Path.cwd())).expanduser().resolve(strict=False)
            result = run_command(render_command(command, variables), cwd=worktree, timeout=int(reader.get("smoke_timeout_seconds") or 120))
            result["name"] = reader.get("name")
            result["ok"] = result.get("rc") == 0
            ok = ok and bool(result["ok"])
            reader_results.append(result)
    internal_results = run_internal_smoke(db_path, db_path.parent, cfg.tenant_id, tuple(dict(x) for x in cfg.control_customers))
    if internal_results:
        ok = ok and all(item["ok"] for item in internal_results)
    report.update(
        {
            "snapshot_db": str(db_path),
            "quick_check": quick_check(db_path),
            "reader_results": reader_results,
            "internal_control_customers": internal_results,
            "status": "ok" if ok else "failed",
        }
    )
    return report, ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test Customer Timeline snapshot through reader APIs.")
    add_common_args(parser)
    parser.add_argument("--snapshot-db", type=Path, required=True)
    args = parser.parse_args()
    report, ok = smoke(args.config, snapshot_db=args.snapshot_db)
    return finish_cli(report, args.out, ok=ok)


if __name__ == "__main__":
    raise SystemExit(main())
