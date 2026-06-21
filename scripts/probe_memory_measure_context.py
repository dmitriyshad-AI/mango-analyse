#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mango_mvp.channels.subscription_llm_parts.direct_path import _direct_path_bot_safe_context_prompt_block
from scripts.run_telegram_dynamic_client_sim import (
    build_dynamic_bot_safe_crm_context,
    load_dynamic_sim_input,
)


DEFAULT_SCENARIOS = Path("product_data/telegram_dynamic_test_sets/memory_rich_2026-06-21.jsonl")
REPO_RELATIVE_TIMELINE_DB = Path("product_data/customer_timeline/customer_timeline_prod_20260621/customer_timeline.sqlite")
MAIN_FOLDER_TIMELINE_DB = Path(
    "/Users/dmitrijfabarisov/Projects/Mango analyse/product_data/customer_timeline/"
    "customer_timeline_prod_20260621/customer_timeline.sqlite"
)
DEFAULT_REPORT = Path("audits/_inbox/memory_measure_apparatus_2026-06-21/context_probe_report.json")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Probe OFF/ON bot-safe context injection without running M1.")
    parser.add_argument("--scenarios", type=Path, default=DEFAULT_SCENARIOS)
    parser.add_argument("--timeline-db", type=Path, default=_default_timeline_db())
    parser.add_argument("--out", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--include-dual", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)

    loaded = load_dynamic_sim_input(args.scenarios)
    personas = _select_probe_personas(loaded.personas, limit=args.limit, include_dual=args.include_dual)
    rows = [probe_persona(persona, timeline_db=args.timeline_db) for persona in personas]
    report = {
        "schema_version": "memory_measure_context_probe_v1",
        "scenarios": str(args.scenarios),
        "timeline_db": str(args.timeline_db),
        "examples": rows,
        "passed": all(row["off_visible_items"] == 0 and row["on_visible_items"] > 0 for row in rows),
        "note": "Probe builds the same read_only_customer_context consumed by direct_path; it does not call LLM or run M1.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"report": str(args.out), "passed": report["passed"], "examples": len(rows)}, ensure_ascii=False))
    return 0 if report["passed"] else 2


def probe_persona(persona: Mapping[str, Any], *, timeline_db: Path) -> Mapping[str, Any]:
    old_flag = os.environ.get("TELEGRAM_BOT_SAFE_CRM_CONTEXT")
    old_db = os.environ.get("TELEGRAM_BOT_SAFE_CRM_CONTEXT_DB")
    try:
        os.environ["TELEGRAM_BOT_SAFE_CRM_CONTEXT_DB"] = str(timeline_db)
        os.environ["TELEGRAM_BOT_SAFE_CRM_CONTEXT"] = "0"
        off_context = build_dynamic_bot_safe_crm_context(persona, active_brand=str(persona.get("brand") or "unknown"))
        off_block = _prompt_block(persona, off_context)
        os.environ["TELEGRAM_BOT_SAFE_CRM_CONTEXT"] = "1"
        on_context = build_dynamic_bot_safe_crm_context(persona, active_brand=str(persona.get("brand") or "unknown"))
        on_block = _prompt_block(persona, on_context)
    finally:
        _restore_env("TELEGRAM_BOT_SAFE_CRM_CONTEXT", old_flag)
        _restore_env("TELEGRAM_BOT_SAFE_CRM_CONTEXT_DB", old_db)
    other_brand = "УНПК" if str(persona.get("brand") or "").casefold() == "foton" else "Фотон"
    return {
        "dialog_id": persona.get("dialog_id"),
        "brand": persona.get("brand"),
        "category": persona.get("category"),
        "customer_id": persona.get("bot_safe_customer_id") or persona.get("customer_id"),
        "off_found": bool(off_context.get("found")) if isinstance(off_context, Mapping) else False,
        "off_visible_items": _visible_item_count(off_block),
        "on_found": bool(on_context.get("found")) if isinstance(on_context, Mapping) else False,
        "on_visible_items": _visible_item_count(on_block),
        "on_prompt_chars": len(on_block),
        "on_preview": on_block[:240],
        "on_item_preview": _first_item_preview(on_block),
        "other_brand_marker_in_on_prompt": bool(other_brand and other_brand in on_block),
    }


def _prompt_block(persona: Mapping[str, Any], crm_context: Mapping[str, Any]) -> str:
    prompt_context = {
        "active_brand": persona.get("brand") or "unknown",
        "read_only_customer_context": crm_context,
    }
    return _direct_path_bot_safe_context_prompt_block(prompt_context)


def _visible_item_count(block: str) -> int:
    return sum(1 for line in block.splitlines() if line.strip()[:2] in {"1.", "2.", "3."})


def _first_item_preview(block: str) -> str:
    for line in block.splitlines():
        value = line.strip()
        if value[:2] in {"1.", "2.", "3."}:
            return value[:240]
    return ""


def _select_probe_personas(
    personas: Sequence[Mapping[str, Any]],
    *,
    limit: int,
    include_dual: bool,
) -> list[Mapping[str, Any]]:
    selected: list[Mapping[str, Any]] = []
    for brand in ("foton", "unpk"):
        for persona in personas:
            if str(persona.get("brand") or "").casefold() == brand and persona.get("category") == "memory_rich":
                selected.append(persona)
                break
    if include_dual:
        for persona in personas:
            if persona.get("category") == "memory_dual_brand_neg" and str(persona.get("brand") or "").casefold() in {"foton", "unpk"}:
                selected.append(persona)
                break
    return selected[: max(1, int(limit or 3))]


def _restore_env(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


def _default_timeline_db() -> Path:
    return REPO_RELATIVE_TIMELINE_DB if REPO_RELATIVE_TIMELINE_DB.exists() else MAIN_FOLDER_TIMELINE_DB


if __name__ == "__main__":
    raise SystemExit(main())
