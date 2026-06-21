from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from mango_mvp.customer_timeline.nightly_incremental import (
    BotSafeRebuildConfig,
    IncrementalSourceConfig,
    NightlyIncrementalConfig,
    ProfileRebuildConfig,
    run_nightly_incremental,
    summarize_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run read-only-source nightly incremental customer_timeline refresh. "
            "The command writes only to the configured customer_timeline SQLite DB and optional derived profile DB."
        )
    )
    parser.add_argument("--config", required=True, help="JSON config with local source list and output paths.")
    parser.add_argument("--summary-only", action="store_true", help="Print compact summary instead of full report.")
    return parser


def config_from_json(path: Path) -> NightlyIncrementalConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("nightly config must be a JSON object")
    sources_payload = payload.get("sources")
    if not isinstance(sources_payload, list) or not sources_payload:
        raise ValueError("nightly config must contain non-empty sources list")
    sources = tuple(source_from_json(item) for item in sources_payload)
    profile_rebuild = None
    if isinstance(payload.get("profile_rebuild"), Mapping):
        profile = payload["profile_rebuild"]
        profile_rebuild = ProfileRebuildConfig(
            profiles_db=Path(str(profile["profiles_db"])),
            master_calls_db=Path(str(profile["master_calls_db"])) if profile.get("master_calls_db") else None,
            build_id=str(profile["build_id"]) if profile.get("build_id") else None,
        )
    bot_safe_rebuild = None
    if isinstance(payload.get("bot_safe_rebuild"), Mapping):
        bot_safe = payload["bot_safe_rebuild"]
        bot_safe_rebuild = BotSafeRebuildConfig(
            allowed_root=Path(str(bot_safe.get("allowed_root") or payload.get("allowed_root") or ".")),
            apply=bool(bot_safe.get("apply")),
            limit=int(bot_safe["limit"]) if bot_safe.get("limit") is not None else None,
        )
    return NightlyIncrementalConfig(
        timeline_db=Path(str(payload["timeline_db"])),
        allowed_root=Path(str(payload.get("allowed_root") or Path(str(payload["timeline_db"])).parent)),
        sources=sources,
        journal_path=Path(str(payload["journal_path"])),
        tenant_id=str(payload.get("tenant_id") or "foton"),
        safety_margin_seconds=int(payload.get("safety_margin_seconds", 300)),
        lock_timeout_seconds=float(payload.get("lock_timeout_seconds", 30.0)),
        actor=str(payload.get("actor") or "customer_timeline_nightly_incremental"),
        profile_rebuild=profile_rebuild,
        bot_safe_rebuild=bot_safe_rebuild,
    )


def source_from_json(payload: Any) -> IncrementalSourceConfig:
    if not isinstance(payload, Mapping):
        raise ValueError("source config item must be an object")
    return IncrementalSourceConfig(
        name=str(payload.get("name") or payload["source_system"]),
        source_system=str(payload["source_system"]),
        path=Path(str(payload["path"])),
        tenant_id=str(payload.get("tenant_id") or "foton"),
        source_ref=str(payload["source_ref"]) if payload.get("source_ref") else None,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_nightly_incremental(config_from_json(Path(args.config)))
    output = summarize_report(report) if args.summary_only else report
    print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

