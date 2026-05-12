from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from mango_mvp.channels.contracts import ChannelAttachment, ChannelDirection, ChannelMessage, ChannelSession
from mango_mvp.channels.preview_service import build_channel_draft_preview
from mango_mvp.customer_timeline.approved_context_pack import (
    APPROVED_CONTEXT_PACK_STATUS,
    CUSTOMER_TIMELINE_APPROVED_CONTEXT_PACK_SCHEMA_VERSION,
)
from mango_mvp.customer_timeline.ids import stable_prefixed_id
from mango_mvp.customer_timeline.safety import blocked_live_actions, guard_customer_timeline_output_path


CUSTOMER_TIMELINE_CHANNEL_PREVIEW_FROM_PACK_SCHEMA_VERSION = "customer_timeline_channel_preview_from_pack_v1"
CHANNEL_PREVIEW_FROM_PACK_STATUS_READY = "draft_ready_for_manager_review"
CHANNEL_PREVIEW_FROM_PACK_STATUS_BLOCKED = "blocked"

PRICE_MARKERS = ("цена", "стоимост", "оплат", "сколько", "рассроч", "скидк")
CALLBACK_MARKERS = ("позвон", "перезвон", "свяж", "звонок")
MANAGER_MARKERS = ("менеджер", "оператор", "человек", "живой")
HOT_MARKERS = ("готов оплатить", "готова оплатить", "записаться", "срочно")
FORBIDDEN_PREVIEW_KEYS = {
    "raw_payload",
    "provider_raw_payload",
    "record_json",
    "audio_path",
    "transcript_path",
    "local_path",
    "path",
}
FORBIDDEN_PREVIEW_MARKERS = (
    "raw_payload",
    "provider_raw_payload",
    "record_json",
    "audio_path",
    "transcript_path",
    "/not/read/",
    "/secret/",
)
EMAIL_RE = re.compile(r"(?i)[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}")
PHONE_RE = re.compile(r"(?<!\w)(?:\+\d[\d\s().-]{8,}\d|7[\d\s().-]{9,}\d|8[\d\s().-]{9,}\d)(?!\w)")


@dataclass(frozen=True)
class CustomerTimelineChannelPreviewFromPackConfig:
    allowed_root: Path
    context_pack_json: Path
    inbound_message_json: Path
    out_preview_json: Optional[Path] = None

    def __post_init__(self) -> None:
        root = Path(self.allowed_root).resolve(strict=False)
        context_pack_json = guard_preview_input_path(self.context_pack_json, root)
        inbound_message_json = guard_preview_input_path(self.inbound_message_json, root)
        out_preview_json = guard_preview_output_path(self.out_preview_json, root, context_pack_json, inbound_message_json)
        object.__setattr__(self, "allowed_root", root)
        object.__setattr__(self, "context_pack_json", context_pack_json)
        object.__setattr__(self, "inbound_message_json", inbound_message_json)
        object.__setattr__(self, "out_preview_json", out_preview_json)


def build_customer_timeline_channel_preview_from_pack(
    *,
    config: CustomerTimelineChannelPreviewFromPackConfig,
    generated_at: Optional[datetime] = None,
) -> Mapping[str, Any]:
    generated = generated_at or datetime.now(timezone.utc)
    pack = load_json_object(config.context_pack_json, "approved context pack JSON")
    inbound_payload = load_json_object(config.inbound_message_json, "inbound message JSON")
    message = channel_message_from_payload(inbound_payload, default_received_at=generated)
    blockers = context_pack_blockers(pack)
    if message.direction != ChannelDirection.INBOUND:
        blockers.append("message_direction_not_inbound")

    context = build_preview_context_from_pack(pack, message)
    preview_payload: Optional[Mapping[str, Any]] = None
    if not blockers:
        session = ChannelSession.from_message(
            message,
            normalized_customer_id=str(pack.get("customer_id") or ""),
            state={
                "approved_context_pack_id": pack.get("pack_id"),
                "approved_context_pack_status": pack.get("status"),
                "preview_mode": "manager_review_only",
            },
            context_summary=str((pack.get("channel_context") or {}).get("safe_context_summary") or ""),
            updated_at=message.received_at,
        )
        preview = build_channel_draft_preview(message, session=session, context=context)
        preview_payload = scrub_preview_payload(preview.to_json_dict(include_raw_payload=False))
        preview_payload = enrich_preview_payload(preview_payload, pack, context, generated_at=generated)

    status = CHANNEL_PREVIEW_FROM_PACK_STATUS_READY if not blockers else CHANNEL_PREVIEW_FROM_PACK_STATUS_BLOCKED
    report = {
        "schema_version": CUSTOMER_TIMELINE_CHANNEL_PREVIEW_FROM_PACK_SCHEMA_VERSION,
        "artifact": "customer_timeline_channel_preview_from_pack",
        "generated_at": generated.isoformat(),
        "validation_ok": not blockers,
        "status": status,
        "preview_id": stable_preview_report_id(pack, message),
        "summary": {
            "validation_ok": not blockers,
            "status": status,
            "blocked_reasons": stable_unique(blockers),
            "draft_created": preview_payload is not None,
            "requires_manager_review": True,
            "live_actions_available": False,
            "can_send": False,
            "context_chunks": int(((pack.get("summary") or {}).get("context_chunks") or 0)) if isinstance(pack.get("summary"), Mapping) else 0,
        },
        "source_refs": {
            "context_pack_sha256": file_sha256(config.context_pack_json),
            "context_pack_id": pack.get("pack_id"),
            "context_pack_schema_version": pack.get("schema_version"),
            "context_pack_status": pack.get("status"),
            "message_sha256": file_sha256(config.inbound_message_json),
            "message_idempotency_key": message.idempotency_key,
        },
        "input_message": project_message_summary(message),
        "context_policy": {
            "uses_approved_customer_context": not blockers,
            "draft_only": True,
            "manager_review_required": True,
            "live_send": False,
            "write_crm": False,
            "write_tallanto": False,
            "write_runtime_db": False,
        },
        "draft_preview": preview_payload,
        "safety": customer_timeline_channel_preview_from_pack_safety_contract(),
    }
    leaked_markers = forbidden_preview_markers(report)
    if leaked_markers:
        report["validation_ok"] = False
        report["status"] = CHANNEL_PREVIEW_FROM_PACK_STATUS_BLOCKED
        report["summary"]["validation_ok"] = False
        report["summary"]["status"] = CHANNEL_PREVIEW_FROM_PACK_STATUS_BLOCKED
        report["summary"]["blocked_reasons"] = stable_unique(
            list(report["summary"]["blocked_reasons"]) + [f"forbidden_marker:{marker}" for marker in leaked_markers]
        )
        report["summary"]["draft_created"] = False
        report["draft_preview"] = None
    if config.out_preview_json:
        config.out_preview_json.parent.mkdir(parents=True, exist_ok=True)
        config.out_preview_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def context_pack_blockers(pack: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    if pack.get("schema_version") != CUSTOMER_TIMELINE_APPROVED_CONTEXT_PACK_SCHEMA_VERSION:
        raise ValueError("context pack schema_version is not customer_timeline_approved_context_pack_v1")
    if pack.get("artifact") != "customer_timeline_approved_context_pack":
        raise ValueError("context pack artifact is not customer_timeline_approved_context_pack")
    summary = pack.get("summary") if isinstance(pack.get("summary"), Mapping) else {}
    channel_context = pack.get("channel_context") if isinstance(pack.get("channel_context"), Mapping) else {}
    approved_context = pack.get("approved_context") if isinstance(pack.get("approved_context"), Mapping) else {}
    if not str(pack.get("pack_id") or "").strip():
        blockers.append("context_pack_id_required")
    if not str(pack.get("tenant_id") or "").strip():
        blockers.append("context_pack_tenant_id_required")
    if not str(pack.get("customer_id") or "").strip():
        blockers.append("context_pack_customer_id_required")
    if pack.get("validation_ok") is not True:
        blockers.append("context_pack_not_valid")
    if pack.get("status") != APPROVED_CONTEXT_PACK_STATUS:
        blockers.append(f"context_pack_not_approved:{pack.get('status') or 'unknown'}")
    if summary.get("validation_ok") is not True:
        blockers.append("context_pack_summary_not_valid")
    if summary.get("blocked_reasons"):
        blockers.append("context_pack_has_blocked_reasons")
    if int(summary.get("context_chunks") or 0) <= 0:
        blockers.append("context_pack_has_no_context_chunks")
    if summary.get("live_actions_available") is not False:
        blockers.append("context_pack_must_not_enable_live_actions")
    if channel_context.get("can_build_draft") is not True:
        blockers.append("context_pack_cannot_build_draft")
    if channel_context.get("can_send") is not False:
        blockers.append("context_pack_must_not_allow_send")
    if channel_context.get("requires_manager_approval_before_send") is not True:
        blockers.append("context_pack_must_require_manager_approval")
    if channel_context.get("normalized_customer_id") != pack.get("customer_id"):
        blockers.append("context_pack_channel_customer_mismatch")
    if not approved_context_items_are_safe(approved_context.get("items") if isinstance(approved_context.get("items"), list) else []):
        blockers.append("context_pack_approved_items_not_safe")
    blockers.extend(validate_preview_safety_snapshot(pack.get("safety"), prefix="context_pack_safety"))
    if forbidden_preview_markers(pack):
        blockers.append("context_pack_contains_forbidden_markers")
    return stable_unique(blockers)


def build_preview_context_from_pack(pack: Mapping[str, Any], message: ChannelMessage) -> Mapping[str, Any]:
    channel_context = pack.get("channel_context") if isinstance(pack.get("channel_context"), Mapping) else {}
    summary = str(channel_context.get("safe_context_summary") or "").strip()
    message_text = message.text.casefold()
    summary_text = summary.casefold()
    commercial = has_marker(message_text, PRICE_MARKERS) or has_marker(summary_text, PRICE_MARKERS)
    callback = has_marker(message_text, CALLBACK_MARKERS) or has_marker(summary_text, CALLBACK_MARKERS)
    manager = has_marker(message_text, MANAGER_MARKERS)
    hot = has_marker(message_text, HOT_MARKERS)
    return {
        "safe_draft_text": build_safe_draft_text_from_context(message, context_summary=summary, commercial=commercial, callback=callback),
        "approved_context_pack_id": pack.get("pack_id"),
        "approved_context_pack_status": pack.get("status"),
        "normalized_customer_id": pack.get("customer_id"),
        "safe_context_summary": summary,
        "context_chunks_count": int(((pack.get("summary") or {}).get("context_chunks") or 0)) if isinstance(pack.get("summary"), Mapping) else 0,
        "requires_commercial_review": commercial,
        "force_manager_handoff": manager,
        "lead_priority": "hot" if hot else "",
        "draft_only": True,
        "live_send_enabled": False,
    }


def build_safe_draft_text_from_context(
    message: ChannelMessage,
    *,
    context_summary: str,
    commercial: bool,
    callback: bool,
) -> str:
    if message.attachments and not message.text:
        return "Здравствуйте! Получили вложение. Менеджер проверит детали и вернется с ответом."
    if commercial:
        return "Здравствуйте! Спасибо за вопрос. Менеджер уточнит детали по стоимости и формату обучения и вернется с ответом."
    if callback:
        return "Здравствуйте! Спасибо, сообщение получили. Передадим менеджеру, он свяжется с вами по вашему вопросу."
    if context_summary:
        return "Здравствуйте! Спасибо за сообщение. Менеджер посмотрит историю вашего обращения и вернется с ответом."
    return "Здравствуйте! Спасибо за сообщение. Менеджер уточнит детали и вернется с ответом."


def channel_message_from_payload(payload: Mapping[str, Any], *, default_received_at: datetime) -> ChannelMessage:
    attachments = tuple(channel_attachment_from_payload(item) for item in payload.get("attachments") or ())
    return ChannelMessage(
        channel=str(payload.get("channel") or "site_chat"),
        channel_message_id=str(payload.get("channel_message_id") or payload.get("message_id") or ""),
        channel_thread_id=str(payload.get("channel_thread_id") or payload.get("thread_id") or ""),
        channel_user_id=str(payload.get("channel_user_id") or payload.get("user_id") or ""),
        direction=payload.get("direction") or "inbound",
        text=str(payload.get("text") or ""),
        received_at=parse_datetime(payload.get("received_at"), default_received_at),
        attachments=attachments,
        raw_payload={},
        metadata=safe_mapping(payload.get("metadata")),
    )


def channel_attachment_from_payload(payload: Mapping[str, Any]) -> ChannelAttachment:
    if not isinstance(payload, Mapping):
        raise ValueError("message attachments must be objects")
    return ChannelAttachment(
        kind=str(payload.get("kind") or "document"),
        uri=str(payload.get("uri") or "memory://attachment"),
        content_type=payload.get("content_type"),
        size_bytes=payload.get("size_bytes"),
        metadata=safe_mapping(payload.get("metadata")),
    )


def enrich_preview_payload(
    payload: Mapping[str, Any],
    pack: Mapping[str, Any],
    context: Mapping[str, Any],
    *,
    generated_at: datetime,
) -> Mapping[str, Any]:
    result = dict(payload)
    result["created_at"] = generated_at.isoformat()
    base_draft_id = str(result.get("draft_id") or "")
    base_preview_key = str(result.get("idempotency_key") or "")
    contextual_draft_id = stable_prefixed_id(
        "channel_pack_draft",
        {"base_draft_id": base_draft_id, "pack_id": pack.get("pack_id")},
        length=32,
    )
    contextual_preview_key = stable_prefixed_id(
        "channel_pack_preview",
        {"base_preview_key": base_preview_key, "pack_id": pack.get("pack_id")},
        length=32,
    )
    result["draft_id"] = contextual_draft_id
    result["idempotency_key"] = contextual_preview_key
    reply = dict(result.get("reply") or {})
    metadata = dict(reply.get("metadata") or {})
    metadata.update(
        {
            "approved_context_pack_id": pack.get("pack_id"),
            "approved_context_pack_schema_version": pack.get("schema_version"),
            "customer_context_used": True,
            "draft_only": True,
            "live_send_enabled": False,
        }
    )
    metadata["draft_id"] = contextual_draft_id
    reply["metadata"] = metadata
    actions = []
    for action in reply.get("recommended_actions") or []:
        if not isinstance(action, Mapping):
            continue
        projected = dict(action)
        action_payload = dict(projected.get("payload") or {})
        action_payload["draft_id"] = contextual_draft_id
        action_payload["approved_context_pack_id"] = pack.get("pack_id")
        action_payload["context_pack_required"] = True
        action_payload["live_send_enabled"] = False
        projected["payload"] = action_payload
        projected["idempotency_key"] = stable_prefixed_id(
            "recommended_action",
            {
                "base_idempotency_key": action.get("idempotency_key"),
                "pack_id": pack.get("pack_id"),
                "action_type": action.get("action_type"),
            },
            length=32,
        )
        actions.append(projected)
    reply["recommended_actions"] = actions
    result["reply"] = reply
    result["approved_context"] = {
        "pack_id": pack.get("pack_id"),
        "customer_id": pack.get("customer_id"),
        "context_chunks": context.get("context_chunks_count"),
        "safe_context_summary": redact_text(str(context.get("safe_context_summary") or "")),
    }
    return result


def approved_context_items_are_safe(items: Sequence[Any]) -> bool:
    if not items:
        return False
    for item in items:
        if not isinstance(item, Mapping):
            return False
        if item.get("allowed_for_bot") is not True:
            return False
        if item.get("requires_manager_review") is not False:
            return False
        if item.get("customer_id") is not None or item.get("opportunity_id") is not None or item.get("event_id") is not None:
            return False
        if forbidden_preview_markers(item):
            return False
    return True


def project_message_summary(message: ChannelMessage) -> Mapping[str, Any]:
    return {
        "channel": message.channel,
        "channel_message_id": redact_text(message.channel_message_id),
        "channel_thread_id": redact_text(message.channel_thread_id),
        "channel_user_id": redact_text(message.channel_user_id),
        "direction": message.direction.value,
        "received_at": message.received_at.isoformat(),
        "text": redact_text(message.text),
        "attachments": len(message.attachments),
        "idempotency_key": message.idempotency_key,
    }


def scrub_preview_payload(value: Any) -> Any:
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            normalized = str(key).casefold()
            if normalized in FORBIDDEN_PREVIEW_KEYS:
                continue
            if normalized == "uri" and isinstance(item, str) and looks_like_local_path(item):
                result[str(key)] = "<attachment_uri_removed>"
            else:
                result[str(key)] = scrub_preview_payload(item)
        return result
    if isinstance(value, list):
        return [scrub_preview_payload(item) for item in value]
    if isinstance(value, tuple):
        return [scrub_preview_payload(item) for item in value]
    if isinstance(value, str):
        return redact_text(value)
    return value


def redact_text(value: str) -> str:
    text = EMAIL_RE.sub("<email_masked>", str(value or ""))
    text = PHONE_RE.sub("<phone_masked>", text)
    if looks_like_local_path(text):
        return "<local_path_removed>"
    return text


def looks_like_local_path(value: str) -> bool:
    text = str(value or "").strip()
    if text.startswith(("file://", "/Users/", "/private/", "/tmp/", "/var/", "./", "../")):
        return True
    return any(marker in text for marker in ("/not/read/", "/secret/", "stable_runtime/"))


def forbidden_preview_markers(payload: Mapping[str, Any]) -> list[str]:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True).casefold()
    return [marker for marker in FORBIDDEN_PREVIEW_MARKERS if marker.casefold() in text]


def validate_preview_safety_snapshot(value: Any, *, prefix: str) -> list[str]:
    if not isinstance(value, Mapping):
        return [f"{prefix}_required"]
    blockers: list[str] = []
    for action in blocked_live_actions():
        if value.get(action) is not False:
            blockers.append(f"{prefix}_{action}_must_be_false")
    for action in ("network_calls", "subprocess_calls", "write_product_timeline_db", "llm_calls", "rag_used"):
        if value.get(action) is not False:
            blockers.append(f"{prefix}_{action}_must_be_false")
    return blockers


def customer_timeline_channel_preview_from_pack_safety_contract() -> Mapping[str, Any]:
    return {
        "schema_version": CUSTOMER_TIMELINE_CHANNEL_PREVIEW_FROM_PACK_SCHEMA_VERSION,
        "read_only_context": True,
        "draft_only": True,
        "write_preview_artifact": True,
        "write_product_timeline_db": False,
        "write_crm": False,
        "write_tallanto": False,
        "send_email": False,
        "send_messenger": False,
        "live_send": False,
        "run_asr": False,
        "run_ra": False,
        "write_runtime_db": False,
        "runtime_db_writes": False,
        "mutate_stable_runtime": False,
        "stable_runtime_writes": False,
        "delete_source_artifacts": False,
        "network_calls": False,
        "subprocess_calls": False,
        "llm_calls": False,
        "rag_used": False,
    }


def stable_preview_report_id(pack: Mapping[str, Any], message: ChannelMessage) -> str:
    return stable_prefixed_id(
        "channel_preview_from_pack",
        {
            "pack_id": pack.get("pack_id"),
            "message_key": message.idempotency_key,
        },
        length=24,
    )


def parse_datetime(value: Any, default: datetime) -> datetime:
    if value is None or not str(value).strip():
        return default
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("received_at must be timezone-aware")
    return parsed


def load_json_object(path: Path, label: str) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} root must be an object")
    return payload


def safe_mapping(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    return {
        str(key): scrub_preview_payload(item)
        for key, item in value.items()
        if str(key).casefold() not in FORBIDDEN_PREVIEW_KEYS
    }


def has_marker(text: str, markers: Sequence[str]) -> bool:
    return any(marker in text for marker in markers)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_unique(items: Sequence[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item not in seen:
            result.append(item)
            seen.add(item)
    return result


def guard_preview_input_path(path: Path | str, allowed_root: Path) -> Path:
    resolved = guard_customer_timeline_output_path(path, allowed_root)
    if not resolved.exists():
        raise ValueError(f"channel preview input does not exist: {resolved}")
    if resolved.is_dir():
        raise ValueError(f"channel preview input must be a file: {resolved}")
    return resolved


def guard_preview_output_path(
    path: Optional[Path],
    allowed_root: Path,
    context_pack_json: Path,
    inbound_message_json: Path,
) -> Optional[Path]:
    if path is None:
        return None
    resolved = guard_customer_timeline_output_path(path, allowed_root)
    if resolved == context_pack_json:
        raise ValueError("channel preview output must not overwrite context pack JSON")
    if resolved == inbound_message_json:
        raise ValueError("channel preview output must not overwrite inbound message JSON")
    return resolved


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = CustomerTimelineChannelPreviewFromPackConfig(
            allowed_root=Path(args.allowed_root),
            context_pack_json=Path(args.context_pack_json),
            inbound_message_json=Path(args.inbound_message_json),
            out_preview_json=Path(args.out_preview_json) if args.out_preview_json else None,
        )
        report = build_customer_timeline_channel_preview_from_pack(config=config)
        if not args.out_preview_json:
            print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if report["validation_ok"] else 1
    except Exception as exc:  # noqa: BLE001 - CLI-facing compact error.
        print(f"customer timeline channel preview from pack failed: {exc}", file=sys.stderr)
        return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a manager-reviewed channel draft from an approved Customer Timeline context pack.")
    parser.add_argument("--allowed-root", required=True)
    parser.add_argument("--context-pack-json", required=True)
    parser.add_argument("--inbound-message-json", required=True)
    parser.add_argument("--out-preview-json")
    return parser


__all__ = [
    "CHANNEL_PREVIEW_FROM_PACK_STATUS_BLOCKED",
    "CHANNEL_PREVIEW_FROM_PACK_STATUS_READY",
    "CUSTOMER_TIMELINE_CHANNEL_PREVIEW_FROM_PACK_SCHEMA_VERSION",
    "CustomerTimelineChannelPreviewFromPackConfig",
    "build_customer_timeline_channel_preview_from_pack",
    "channel_message_from_payload",
    "customer_timeline_channel_preview_from_pack_safety_contract",
    "main",
]
