#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

ENV_FILES = (
    ROOT / "stable_runtime" / "amocrm_runtime" / ".env.private",
    ROOT / "prod_runtime_transfer" / ".env.private",
)
DEFAULT_TIMELINE_DB = Path(
    "/Users/dmitrijfabarisov/Projects/Mango_tz139_customer_timeline/product_data/customer_timeline/"
    "canonical_readonly_20260621_with_channels/customer_timeline.sqlite"
)
DEFAULT_ALLOWED_ROOT = DEFAULT_TIMELINE_DB.parent


def _load_env_files_early() -> None:
    for path in ENV_FILES:
        if not path.exists():
            continue
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())
    os.environ.setdefault(
        "DATABASE_URL",
        f"sqlite:///{(ROOT / 'stable_runtime' / 'amocrm_runtime' / 'amo_runtime.db').resolve()}",
    )


_load_env_files_early()

from mango_mvp.crm_card_amo_writeback import (  # noqa: E402
    LIVE_CONFIRMATION,
    CrmCardAmoDryRunConfig,
    amo_client_from_context,
    build_crm_card_amo_dry_run,
    resolve_amo_access_context_no_refresh,
)
from mango_mvp.crm_card_history_summary import CrmHistorySummaryConfig, CrmHistorySummarizer  # noqa: E402
from mango_mvp.amocrm_runtime.db import SessionLocal  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dry-run CRM card writeback to AMO contact and deal fields.")
    parser.add_argument("--timeline-db", default=str(DEFAULT_TIMELINE_DB))
    parser.add_argument("--allowed-root", default=str(DEFAULT_ALLOWED_ROOT))
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--tenant-id", default="foton")
    parser.add_argument("--sample-size", type=int, default=5)
    parser.add_argument("--customer-id", action="append", default=[])
    parser.add_argument("--history-summary-provider", default="rule", choices=("off", "rule", "codex_cli"))
    parser.add_argument("--history-summary-cache-dir", default="")
    parser.add_argument("--history-summary-model", default="gpt-5.4-mini")
    parser.add_argument("--history-summary-reasoning", default="low")
    parser.add_argument("--execute-live-write", action="store_true")
    parser.add_argument("--live-confirmation", default="")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.execute_live_write:
        if args.live_confirmation != LIVE_CONFIRMATION:
            print(f"Refusing live AMO writeback: --live-confirmation must be {LIVE_CONFIRMATION!r}.", file=sys.stderr)
            return 2
        print("Refusing live AMO writeback: this TZ A command is dry-run only until Dmitry gives a separate live 'да'.", file=sys.stderr)
        return 2
    if args.live_confirmation:
        print("Refusing live AMO writeback: --live-confirmation is valid only with --execute-live-write.", file=sys.stderr)
        return 2

    timeline_db = Path(args.timeline_db).expanduser().resolve(strict=False)
    allowed_root = Path(args.allowed_root).expanduser().resolve(strict=False)
    out_dir = Path(args.out_dir).expanduser().resolve(strict=False) if args.out_dir else (
        ROOT
        / "audits"
        / "_inbox"
        / f"tzA_crm_card_amo_dry_run_{datetime.now(timezone.utc):%Y%m%d%H%M%S}"
    )
    cache_dir = Path(args.history_summary_cache_dir).expanduser().resolve(strict=False) if args.history_summary_cache_dir else (
        out_dir / "history_summary_cache"
    )
    history_summarizer = CrmHistorySummarizer(
        CrmHistorySummaryConfig(
            provider=args.history_summary_provider,
            cache_dir=cache_dir,
            model=args.history_summary_model,
            reasoning_effort=args.history_summary_reasoning,
        )
    )
    session = SessionLocal()
    try:
        try:
            context = resolve_amo_access_context_no_refresh(session)
            amo_client = amo_client_from_context(context)
            summary = build_crm_card_amo_dry_run(
                CrmCardAmoDryRunConfig(
                    timeline_db=timeline_db,
                    allowed_root=allowed_root,
                    out_dir=out_dir,
                    tenant_id=args.tenant_id,
                    sample_size=args.sample_size,
                    customer_ids=tuple(args.customer_id or ()),
                    history_summarizer=history_summarizer,
                ),
                amo_client=amo_client,
            )
        except Exception as exc:  # noqa: BLE001 - this is a dry-run report, not a live operation.
            out_dir.mkdir(parents=True, exist_ok=True)
            summary = {
                "schema_version": "crm_card_amo_writeback_dry_run_blocked_v1",
                "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "status": "blocked",
                "reason": str(exc),
                "timeline_db": str(timeline_db),
                "allowed_root": str(allowed_root),
                "tenant_id": args.tenant_id,
                "outputs": {"summary_json": str(out_dir / "summary.json")},
                "safety": {
                    "dry_run_only": True,
                    "write_amo": False,
                    "write_tallanto": False,
                    "send_messages": False,
                    "refresh_oauth_token": False,
                    "patch_function_available": False,
                    "write_stable_runtime": False,
                },
            }
            (out_dir / "summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
            return 2
    finally:
        session.rollback()
        session.close()
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
