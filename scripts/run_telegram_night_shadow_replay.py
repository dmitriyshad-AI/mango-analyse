#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from mango_mvp.channels.night_funnel_shadow import (
    DEFAULT_CONTROL_PATH,
    DEFAULT_INBOUND_TEE_PATH,
    DEFAULT_LEAD_STORE_PATH,
    DEFAULT_REPLAY_CURSOR_PATH,
    DEFAULT_SHADOW_LOG_PATH,
    DEFAULT_STATUS_PATH,
    MANAGER_QUEUE,
    NightFunnelControl,
    append_lead_card,
    append_shadow_log,
    brand_from_channel,
    build_lead_card,
    build_shadow_record,
    evaluate_night_gate,
    load_bot_control,
    read_unprocessed_tee_records,
    rotate_inbound_tee,
    save_replay_cursor,
    write_bot_status,
)
from mango_mvp.channels.subscription_llm import AUTONOMY_MATRIX_SAFE_TOPIC_IDS, SubscriptionDraftResult, SubscriptionLlmDraftProvider
from mango_mvp.channels.telegram_pilot_context_builder import build_telegram_pilot_context_from_snapshot


DEFAULT_SNAPSHOT = Path("product_data/knowledge_base/kb_release_20260520_v6_3_team_answers/kb_release_v3_snapshot.json")


def build_replay_context(record: Mapping[str, Any], *, snapshot_path: Path) -> Mapping[str, Any]:
    brand = str(record.get("brand") or "").casefold().strip()
    text = str(record.get("text") or "")
    known_context = record.get("known_context") if isinstance(record.get("known_context"), Mapping) else {}
    known_slots = dict(known_context.get("known_slots") or {}) if isinstance(known_context, Mapping) else {}
    recent_messages = tuple(known_context.get("recent_messages") or ()) if isinstance(known_context, Mapping) else ()
    rop_policy = {
        "bot_permission": "bot_answer_self_for_pilot",
        "autonomy_policy": {
            "allow_autonomous": True,
            "allowed_topic_ids": sorted(AUTONOMY_MATRIX_SAFE_TOPIC_IDS),
            "default": "draft_for_manager_or_manager_only",
            "fact_requirement": "client_safe_fact_verified",
            "p0_overrides_autonomy": True,
        },
    }
    pilot_context = build_telegram_pilot_context_from_snapshot(
        text,
        snapshot_path=snapshot_path,
        active_brand=brand,
        rop_policy=rop_policy,
        recent_messages=recent_messages,
        client_identity={
            "channel": "telegram_night_shadow_replay",
            "channel_thread_id": str(record.get("chat_id_hash") or ""),
            "channel_user_id": str(record.get("chat_id_hash") or ""),
        },
        known_slots=known_slots,
        session_id=f"telegram_night_shadow_replay:{brand}:{record.get('chat_id_hash') or 'unknown'}",
    )
    payload = dict(pilot_context.to_prompt_context())
    payload["active_brand"] = brand
    payload["known_dialog_fields"] = known_slots
    payload["known_slots"] = known_slots
    payload["recent_messages"] = recent_messages
    payload["TELEGRAM_DIALOGUE_CONTRACT_PIPELINE"] = "1"
    payload["night_shadow_replay_mode"] = {
        "enabled": True,
        "shadow_only": True,
        "no_telegram_token": True,
        "no_crm_tallanto_write": True,
    }
    return payload


def replay_tee_records(
    *,
    tee_path: Path,
    cursor_path: Path,
    snapshot_path: Path,
    shadow_log_path: Path,
    lead_store_path: Path,
    status_path: Path,
    control_path: Path = DEFAULT_CONTROL_PATH,
    max_records: int | None = None,
    provider: Any | None = None,
    rotate_retention_days: int | None = None,
) -> Mapping[str, Any]:
    records, cursor = read_unprocessed_tee_records(tee_path, cursor_path, max_records=max_records)
    control = load_bot_control(control_path)
    control = NightFunnelControl(
        enabled=control.enabled,
        mode=control.mode,
        shadow_only=True,
        manual_kill_switch=control.manual_kill_switch,
        night_limit=control.night_limit,
        auto_trip_hold_rate=control.auto_trip_hold_rate,
        auto_trip_error_count=control.auto_trip_error_count,
        morning_followup_hour=control.morning_followup_hour,
        morning_followup_process_confirmed=control.morning_followup_process_confirmed,
    )
    provider = provider or SubscriptionLlmDraftProvider(cache_dir=Path(".codex_local/telegram_night_funnel/llm_cache"))
    decisions: list[Mapping[str, Any]] = []
    processed = 0
    skipped = 0
    brands: set[str] = set()
    for record in records:
        text = str(record.get("text") or "").strip()
        brand = str(record.get("brand") or "").casefold().strip()
        if not text or brand not in {"foton", "unpk"}:
            skipped += 1
            continue
        context = build_replay_context(record, snapshot_path=snapshot_path)
        result: SubscriptionDraftResult = provider.build_draft(text, context=context)
        draft_text = result.draft_text
        pipeline = result.metadata.get("dialogue_contract_pipeline") if isinstance(result.metadata, Mapping) else {}
        retrieved_facts = pipeline.get("retrieved_facts") if isinstance(pipeline, Mapping) and isinstance(pipeline.get("retrieved_facts"), Mapping) else {}
        gate = evaluate_night_gate(
            client_text=text,
            draft_text=draft_text,
            route=result.route,
            active_brand=brand,
            snapshot_path=snapshot_path,
            retrieved_facts=retrieved_facts,
            safety_flags=result.safety_flags,
            control=control,
            prior_decisions=tuple(decisions),
        )
        channel_source = str(record.get("channel_source") or "")
        channel_brand = brand_from_channel(channel_source)
        if channel_brand and channel_brand != brand:
            gate = {
                **dict(gate),
                "decision": MANAGER_QUEUE,
                "reason": f"channel_brand_mismatch:{channel_brand}!={brand}",
                "shadow_only": True,
            }
        record_context = {"known_dialog_fields": (record.get("known_context") or {}).get("known_slots") or {}}
        shadow = build_shadow_record(
            brand=brand,
            channel_source=channel_source,
            utm=record.get("utm") if isinstance(record.get("utm"), Mapping) else {},
            client_text=text,
            draft_text=draft_text,
            gate=gate,
            context=record_context,
        )
        shadow = {**dict(shadow), "tee_record_id": record.get("tee_record_id"), "tee_byte_offset": record.get("tee_byte_offset")}
        lead_card = build_lead_card(
            brand=brand,
            utm=record.get("utm") if isinstance(record.get("utm"), Mapping) else {},
            client_text=text,
            draft_text=draft_text,
            decision=str(gate.get("decision") or ""),
            reason=str(gate.get("reason") or ""),
            context=record_context,
        )
        append_shadow_log(shadow_log_path, shadow)
        append_lead_card(lead_store_path, lead_card)
        decisions.append({"decision": gate.get("decision"), "reason": gate.get("reason")})
        brands.add(brand)
        processed += 1
    save_replay_cursor(cursor_path, cursor)
    write_bot_status(
        status_path,
        brand=next(iter(brands)) if len(brands) == 1 else "mixed",
        control=control,
        decisions=tuple(decisions),
        auto_tripped=any(bool(item.get("auto_tripped")) for item in decisions),
    )
    rotation: Mapping[str, Any] | None = None
    if rotate_retention_days is not None:
        rotation = rotate_inbound_tee(tee_path, retention_days=rotate_retention_days)
    return {
        "ok": True,
        "processed": processed,
        "skipped": skipped,
        "decisions": dict(Counter(str(item.get("decision") or "") for item in decisions)),
        "shadow_log_path": str(shadow_log_path),
        "lead_store_path": str(lead_store_path),
        "status_path": str(status_path),
        "cursor_path": str(cursor_path),
        "rotation": rotation,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Replay passive Telegram night funnel tee records in SHADOW mode.")
    parser.add_argument("--tee-path", type=Path, default=DEFAULT_INBOUND_TEE_PATH)
    parser.add_argument("--cursor-path", type=Path, default=DEFAULT_REPLAY_CURSOR_PATH)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--shadow-log-path", type=Path, default=DEFAULT_SHADOW_LOG_PATH)
    parser.add_argument("--lead-store-path", type=Path, default=DEFAULT_LEAD_STORE_PATH)
    parser.add_argument("--status-path", type=Path, default=DEFAULT_STATUS_PATH)
    parser.add_argument("--control-path", type=Path, default=DEFAULT_CONTROL_PATH)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--reasoning", default="high")
    parser.add_argument("--timeout-sec", type=int, default=240)
    parser.add_argument("--cache-dir", type=Path, default=Path(".codex_local/telegram_night_funnel/llm_cache"))
    parser.add_argument("--rotate-retention-days", type=int, default=None)
    args = parser.parse_args(argv)
    provider = SubscriptionLlmDraftProvider(
        model=args.model,
        reasoning_effort=args.reasoning,
        timeout_sec=args.timeout_sec,
        cache_dir=args.cache_dir,
    )
    summary = replay_tee_records(
        tee_path=args.tee_path,
        cursor_path=args.cursor_path,
        snapshot_path=args.snapshot,
        shadow_log_path=args.shadow_log_path,
        lead_store_path=args.lead_store_path,
        status_path=args.status_path,
        control_path=args.control_path,
        max_records=args.max_records,
        provider=provider,
        rotate_retention_days=args.rotate_retention_days,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
