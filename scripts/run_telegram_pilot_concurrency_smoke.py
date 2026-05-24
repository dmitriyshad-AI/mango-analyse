#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from mango_mvp.channels.contracts import ChannelDirection, ChannelMessage
from mango_mvp.channels.subscription_llm import (
    SubscriptionDraftResult,
    SubscriptionLlmDraftProvider,
    normalize_subscription_draft_payload,
    strip_internal_service_markers,
)
from mango_mvp.channels.telegram_pilot_context_builder import build_telegram_pilot_context_from_snapshot
from mango_mvp.channels.telegram_pilot_store import (
    PILOT_DRAFT_STATUS_MANAGER_ONLY,
    PILOT_DRAFT_STATUS_NEEDS_REVIEW,
    TelegramPilotSQLiteStore,
)


DEFAULT_SNAPSHOT = Path("product_data/knowledge_base/kb_release_20260520_v6_3_team_answers/kb_release_v3_snapshot.json")
DEFAULT_OUT_DIR = Path(".codex_local/telegram_pilot/concurrency_smoke")
SCHEMA_VERSION = "telegram_pilot_concurrency_smoke_v1_2026_05_23"

FOTON_MESSAGES = (
    "Здравствуйте! Сколько стоит 5 класс математика в Фотоне?",
    "Можно ли оплатить курс частями?",
    "Есть ли пробное занятие онлайн?",
    "Где проходят очные занятия в Москве?",
    "Подскажите, как записаться на летнюю школу?",
)
UNPK_MESSAGES = (
    "Здравствуйте! Сколько стоит подготовка для 8 класса в УНПК?",
    "Какая платформа для онлайн-занятий?",
    "Есть ли налоговый вычет?",
    "Какой адрес очных занятий в Москве?",
    "Можно ли использовать маткапитал?",
)


class FakeLoadProvider:
    def build_draft(self, client_message: str, *, context: Mapping[str, Any]) -> SubscriptionDraftResult:
        brand = str(context.get("active_brand") or "unpk")
        brand_name = "Фотоне" if brand == "foton" else "УНПК МФТИ"
        return normalize_subscription_draft_payload(
            {
                "message_type": "question",
                "broad_group": "sales",
                "topic_id": "theme:016_program",
                "confidence_theme": 0.82,
                "confidence_group": 0.82,
                "risk_level": "low",
                "route": "bot_answer_self_for_pilot",
                "draft_text": f"Да, помогу сориентироваться по {brand_name}. Подскажите класс ребёнка и формат: онлайн или очно?",
                "safety_flags": [],
            }
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Safe local concurrency smoke for Telegram pilot decision path. Does not send Telegram messages."
    )
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--store-db", type=Path, default=None)
    parser.add_argument("--brand", choices=("all", "foton", "unpk"), default="all")
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--mode", choices=("fake", "codex"), default="fake")
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--reasoning", default="medium")
    parser.add_argument("--timeout-sec", type=int, default=180)
    args = parser.parse_args(argv)

    if args.requests < 1:
        raise ValueError("--requests must be >= 1")
    if args.concurrency < 1:
        raise ValueError("--concurrency must be >= 1")
    if "stable_runtime" in args.out_dir.resolve(strict=False).parts:
        raise ValueError("Refusing to write concurrency smoke outputs under stable_runtime")
    store_db = args.store_db or (args.out_dir / "telegram_pilot_concurrency_smoke.sqlite")
    if "stable_runtime" in store_db.resolve(strict=False).parts:
        raise ValueError("Refusing to write concurrency smoke store under stable_runtime")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    requests = build_requests(args.brand, args.requests)
    started = time.perf_counter()
    rows: list[Mapping[str, Any]] = []
    errors: list[Mapping[str, Any]] = []

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        future_to_request = {
            executor.submit(process_one_request, request, args=args, store_db=store_db): request
            for request in requests
        }
        for future in as_completed(future_to_request):
            request = future_to_request[future]
            try:
                rows.append(future.result())
            except Exception as exc:  # pragma: no cover - exercised in live smoke failures
                errors.append(
                    {
                        "request_id": request["request_id"],
                        "brand": request["brand"],
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )

    rows = sorted(rows, key=lambda item: int(item.get("index") or 0))
    errors = sorted(errors, key=lambda item: str(item.get("request_id") or ""))
    summary = build_summary(rows, errors, args=args, store_db=store_db, elapsed_seconds=time.perf_counter() - started)
    write_jsonl(args.out_dir / "requests.jsonl", rows)
    write_jsonl(args.out_dir / "errors.jsonl", errors)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    (args.out_dir / "summary.md").write_text(render_summary_md(summary), encoding="utf-8")

    print(json.dumps({"ok": not errors, "out_dir": str(args.out_dir), **summary["totals"]}, ensure_ascii=False, indent=2, sort_keys=True))
    return 1 if errors else 0


def build_requests(brand: str, count: int) -> list[Mapping[str, Any]]:
    brands = ("foton", "unpk") if brand == "all" else (brand,)
    rows: list[Mapping[str, Any]] = []
    for index in range(count):
        active_brand = brands[index % len(brands)]
        messages = FOTON_MESSAGES if active_brand == "foton" else UNPK_MESSAGES
        text = messages[index % len(messages)]
        rows.append(
            {
                "index": index,
                "request_id": f"load_{index:04d}_{active_brand}",
                "brand": active_brand,
                "text": text,
                "thread_id": f"load_thread_{index:04d}",
                "user_id": f"load_user_{index:04d}",
            }
        )
    return rows


def process_one_request(request: Mapping[str, Any], *, args: argparse.Namespace, store_db: Path) -> Mapping[str, Any]:
    started = time.perf_counter()
    brand = str(request["brand"])
    text = str(request["text"])
    context = build_context(text, brand=brand, request=request, snapshot_path=args.snapshot)
    provider = build_provider(args, request_id=str(request["request_id"]))
    result = provider.build_draft(text, context=context)
    latency = time.perf_counter() - started
    draft_text = strip_internal_service_markers(str(result.draft_text or "")).strip()
    message = ChannelMessage(
        channel="telegram",
        channel_message_id=str(request["request_id"]),
        channel_thread_id=str(request["thread_id"]),
        channel_user_id=str(request["user_id"]),
        direction=ChannelDirection.INBOUND,
        text=text,
        received_at=datetime.now(timezone.utc),
        metadata={"schema_version": SCHEMA_VERSION, "concurrency_smoke": True, "brand": brand},
    )
    status = PILOT_DRAFT_STATUS_MANAGER_ONLY if result.route == "manager_only" else PILOT_DRAFT_STATUS_NEEDS_REVIEW
    with TelegramPilotSQLiteStore(store_db) as store:
        write_result = store.upsert_message_context_draft(
            message,
            context=context,
            draft_text=draft_text,
            prompt_version=SCHEMA_VERSION,
            knowledge_base_version=str(context.get("knowledge_base_version") or args.snapshot.name or "unknown"),
            status=status,
            topic_id=result.topic_id,
            route=result.route,
            safety_flags=result.safety_flags,
            draft_metadata={
                "schema_version": SCHEMA_VERSION,
                "mode": args.mode,
                "model": args.model if args.mode == "codex" else "fake",
                "reasoning_effort": args.reasoning if args.mode == "codex" else "fake",
                "latency_seconds": round(latency, 3),
            },
            actor="telegram_pilot_concurrency_smoke",
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "index": request["index"],
        "request_id": request["request_id"],
        "brand": brand,
        "text": text,
        "route": result.route,
        "topic_id": result.topic_id,
        "risk_level": result.risk_level,
        "safety_flags": list(result.safety_flags),
        "draft_text": draft_text,
        "latency_seconds": round(latency, 3),
        "message_key": write_result.message_key,
        "draft_id": write_result.draft_id,
        "draft_created": write_result.draft_created,
    }


def build_provider(args: argparse.Namespace, *, request_id: str) -> Any:
    if args.mode == "fake":
        return FakeLoadProvider()
    return SubscriptionLlmDraftProvider(
        model=args.model,
        reasoning_effort=args.reasoning,
        timeout_sec=args.timeout_sec,
        cache_dir=Path(".codex_local/telegram_pilot/concurrency_smoke/llm_cache") / safe_filename(request_id),
    )


def build_context(
    client_message: str,
    *,
    brand: str,
    request: Mapping[str, Any],
    snapshot_path: Path,
) -> Mapping[str, Any]:
    pilot_context = build_telegram_pilot_context_from_snapshot(
        client_message,
        snapshot_path=snapshot_path,
        active_brand=brand,
        rop_policy={
            "bot_permission": "bot_answer_self_for_pilot",
            "autonomy_policy": {
                "allow_autonomous": True,
                "default": "draft_for_manager_or_manager_only",
                "fact_requirement": "client_safe_fact_verified",
                "p0_overrides_autonomy": True,
            },
        },
        recent_messages=(),
        client_identity={
            "channel": "telegram",
            "channel_thread_id": str(request["thread_id"]),
            "channel_user_id": str(request["user_id"]),
        },
        customer_summary="Локальный concurrency smoke. Не раскрывать клиенту.",
    )
    payload = dict(pilot_context.to_prompt_context())
    payload["active_brand"] = brand
    payload["concurrency_smoke"] = {
        "enabled": True,
        "request_id": request["request_id"],
        "do_not_disclose": True,
    }
    return payload


def build_summary(
    rows: Sequence[Mapping[str, Any]],
    errors: Sequence[Mapping[str, Any]],
    *,
    args: argparse.Namespace,
    store_db: Path,
    elapsed_seconds: float,
) -> Mapping[str, Any]:
    latencies = [float(row.get("latency_seconds") or 0) for row in rows]
    routes = Counter(str(row.get("route") or "") for row in rows)
    brands = Counter(str(row.get("brand") or "") for row in rows)
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mode": args.mode,
        "model": args.model if args.mode == "codex" else "fake",
        "reasoning_effort": args.reasoning if args.mode == "codex" else "fake",
        "snapshot_path": str(args.snapshot),
        "store_db": str(store_db),
        "safety": {
            "telegram_messages_sent": False,
            "crm_writes": False,
            "tallanto_writes": False,
            "stable_runtime_writes": False,
        },
        "totals": {
            "requests": args.requests,
            "completed": len(rows),
            "errors": len(errors),
            "concurrency": args.concurrency,
            "elapsed_seconds": round(elapsed_seconds, 3),
            "avg_latency_seconds": round(statistics.mean(latencies), 3) if latencies else None,
            "p50_latency_seconds": percentile(latencies, 50),
            "p95_latency_seconds": percentile(latencies, 95),
        },
        "brands": dict(brands),
        "routes": dict(routes),
    }


def percentile(values: Sequence[float], pct: int) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((pct / 100) * (len(ordered) - 1))))
    return round(float(ordered[index]), 3)


def render_summary_md(summary: Mapping[str, Any]) -> str:
    totals = summary.get("totals") if isinstance(summary.get("totals"), Mapping) else {}
    return "\n".join(
        [
            "# Telegram Pilot Concurrency Smoke",
            "",
            "Это локальный smoke: реальные сообщения в Telegram не отправлялись.",
            "",
            f"- Режим: `{summary.get('mode')}`",
            f"- Запросов: `{totals.get('requests')}`",
            f"- Завершено: `{totals.get('completed')}`",
            f"- Ошибок: `{totals.get('errors')}`",
            f"- Параллельность: `{totals.get('concurrency')}`",
            f"- Общее время, сек: `{totals.get('elapsed_seconds')}`",
            f"- Средняя задержка, сек: `{totals.get('avg_latency_seconds')}`",
            f"- p50, сек: `{totals.get('p50_latency_seconds')}`",
            f"- p95, сек: `{totals.get('p95_latency_seconds')}`",
            f"- Бренды: `{summary.get('brands')}`",
            f"- Маршруты: `{summary.get('routes')}`",
            "",
            "Артефакты:",
            "",
            "- `requests.jsonl` — все обработанные локальные обращения.",
            "- `errors.jsonl` — ошибки, если были.",
            "- `telegram_pilot_concurrency_smoke.sqlite` — локальный store с сообщениями/черновиками.",
            "",
        ]
    )


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def safe_filename(value: str) -> str:
    import re

    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned[:120] or "request"


if __name__ == "__main__":
    raise SystemExit(main())
