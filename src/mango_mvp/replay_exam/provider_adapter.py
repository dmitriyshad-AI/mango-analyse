from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from mango_mvp.channels.subscription_llm import SubscriptionLlmDraftProvider
from mango_mvp.channels.subscription_llm_parts.support import (
    DIRECT_PATH_ENV,
    DIRECT_PATH_PILOT_CONFIG_ENV,
    DIRECT_PATH_PILOT_CONFIG_VERSION,
    DIRECT_PATH_PILOT_PROFILE_DEFAULT_ON_FLAGS,
)
from mango_mvp.pilot_context_assembly import build_pilot_context_payload

from .models import BotReplayResult, ReplayCase, ReplayMessage
from .pseudonymizer import pii_signals


SCRUBBED_ROOT = Path("~/.mango_local/replay_exam/scrubbed").expanduser()
RAW_ROOT = Path("~/.mango_local/replay_exam/raw").expanduser()


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def assert_scrubbed_cases_path(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    root = SCRUBBED_ROOT.resolve()
    if not _is_relative_to(resolved, root):
        raise ValueError(f"real replay cases must stay under {root}")
    return resolved


def assert_real_replay_output_path(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if "stable_runtime" in resolved.parts:
        raise ValueError("real replay output must not be written under stable_runtime")
    if _is_relative_to(resolved, RAW_ROOT.resolve()):
        raise ValueError(f"real replay output must not be written under raw dump root {RAW_ROOT.resolve()}")
    if ".codex_local" in resolved.parts and "runtime" in resolved.parts:
        raise ValueError("real replay output must not be written under local runtime directories")
    return resolved


def assert_real_replay_cases_safe(cases: Sequence[ReplayCase], *, allow_non_chat_only: bool = False) -> None:
    for case in cases:
        if case.brand.casefold() not in {"foton", "unpk"}:
            raise ValueError(f"unsupported replay brand for {case.turn_id}: {case.brand}")
        if not allow_non_chat_only and case.segment != "chat_only":
            raise ValueError(f"real replay only accepts chat_only by default: {case.turn_id} segment={case.segment}")
        signals = pii_signals(
            {
                "dialog_id": case.dialog_id,
                "profile_id": case.profile_id,
                "chat_id": case.chat_id,
                "turn_id": case.turn_id,
                "brand": case.brand,
                "client_message": case.client_message,
                "manager_reference": case.manager_reference,
                "prefix_messages": [message.__dict__ for message in case.prefix_messages],
                "segment": case.segment,
                "metadata": dict(case.metadata),
            }
        )
        if signals:
            raise ValueError(f"PII signals in scrubbed replay case {case.turn_id}: {signals}")


def _history_line(message: ReplayMessage) -> str:
    role = "Ответ" if message.from_me else "Клиент"
    text = " ".join(str(message.text or "").split())
    return f"{role}: {text}" if text else ""


def replay_recent_messages(case: ReplayCase, *, older_summary: str = "") -> tuple[str, ...]:
    messages = [_history_line(message) for message in case.prefix_messages]
    messages = [message for message in messages if message]
    if older_summary.strip():
        messages.insert(0, f"Сводка ранней части диалога: {' '.join(older_summary.split())[:1200]}")
    return tuple(messages[-12:])


def build_replay_provider_context(
    case: ReplayCase,
    runner_context: Mapping[str, object],
    *,
    snapshot_path: Path,
) -> Mapping[str, Any]:
    older_summary = str(runner_context.get("older_summary") or "")
    context = build_pilot_context_payload(
        current_text=case.client_message,
        snapshot_path=snapshot_path,
        active_brand=case.brand.casefold(),
        recent_messages=replay_recent_messages(case, older_summary=older_summary),
        dialogue_memory={},
        session_id=f"wappi_replay:{case.brand.casefold()}:{case.dialog_id}",
        channel="wappi_replay",
        channel_thread_id=f"{case.profile_id}:{case.chat_id}",
        channel_user_id=case.chat_id,
        current_message_id=case.turn_id,
        dialogue_contract_pipeline_enabled=True,
        sends_client_replies=False,
        debug_impersonation_enabled=False,
        crm_context={},
    )
    payload = dict(context)
    payload["replay_exam"] = {
        "enabled": True,
        "dialog_id": case.dialog_id,
        "turn_id": case.turn_id,
        "segment": case.segment,
        "manager_reference_available": bool(case.manager_reference),
        "manager_reference_passed_to_provider": False,
    }
    payload["TELEGRAM_DIRECT_PATH_PILOT_CONFIG"] = "pilot_gold_v1"
    payload["direct_path_pilot_config"] = "pilot_gold_v1"
    payload[DIRECT_PATH_ENV] = "1"
    payload["direct_path_enabled"] = True
    payload[DIRECT_PATH_PILOT_CONFIG_ENV] = DIRECT_PATH_PILOT_CONFIG_VERSION
    payload["direct_path_pilot_config"] = DIRECT_PATH_PILOT_CONFIG_VERSION
    for env_name in DIRECT_PATH_PILOT_PROFILE_DEFAULT_ON_FLAGS:
        payload.setdefault(env_name, "1")
    public_mode = payload.get("public_pilot_mode") if isinstance(payload.get("public_pilot_mode"), Mapping) else {}
    if public_mode.get("sends_client_replies") is not False:
        raise RuntimeError("replay context is not read-only: sends_client_replies must be false")
    if public_mode.get("no_crm_tallanto_write") is not True:
        raise RuntimeError("replay context is not read-only: no_crm_tallanto_write must be true")
    if payload.get("read_only_customer_context"):
        raise RuntimeError("replay context must not include CRM/Tallanto/timeline customer context")
    return payload


class RealReplayDraftProvider:
    def __init__(
        self,
        *,
        snapshot_path: Path,
        draft_provider: SubscriptionLlmDraftProvider | None = None,
        cache_dir: Path | None = None,
    ) -> None:
        self.snapshot_path = snapshot_path.expanduser()
        if not self.snapshot_path.exists():
            raise FileNotFoundError(f"snapshot does not exist: {self.snapshot_path}")
        if cache_dir is not None:
            resolved_cache = cache_dir.expanduser().resolve()
            if "stable_runtime" in resolved_cache.parts:
                raise ValueError("replay provider cache must not be under stable_runtime")
            resolved_cache.mkdir(parents=True, exist_ok=True)
        else:
            resolved_cache = None
        self.draft_provider = draft_provider or SubscriptionLlmDraftProvider(cache_dir=resolved_cache)

    def __call__(self, case: ReplayCase, runner_context: Mapping[str, object]) -> BotReplayResult:
        context = build_replay_provider_context(case, runner_context, snapshot_path=self.snapshot_path)
        result = self.draft_provider.build_draft(case.client_message, context=context)
        metadata = dict(result.metadata)
        if result.raw_response:
            metadata["replay_raw_response"] = result.raw_response
        metadata["replay_provider"] = {
            "mode": "real_subscription_llm",
            "snapshot_path": str(self.snapshot_path),
            "live_writes_allowed": False,
            "manager_reference_passed_to_provider": False,
        }
        return BotReplayResult(
            route=result.route,
            bot_text=result.draft_text,
            safety_flags=tuple(result.safety_flags),
            metadata=metadata,
        )
