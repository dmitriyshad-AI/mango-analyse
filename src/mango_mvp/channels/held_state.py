from __future__ import annotations

"""Append-only semantic state for multi-turn Telegram dialogue.

This is a thin state header above DialogueMemory. It stores only durable semantic
roles that should survive short follow-ups: confirmed format, payment source,
transfer sense, group-topic context, P0 latch and retrieval topic.
"""

from dataclasses import dataclass
from typing import Any, Mapping

from mango_mvp.channels.semantic_roles import MessageRoles
from mango_mvp.channels.text_signals import has_any_marker


HELD_STATE_SCHEMA_VERSION = "held_state_v1_2026_05_25"

GROUP_TOPIC_CUES = ("групп", "уровень", "тестир", "распределен", "сильнее", "послабее", "посильнее", "слабее")


@dataclass(frozen=True)
class HeldState:
    training_format: str = ""
    payment_source: str = ""
    transfer_sense: str = ""
    group_topic_active: bool = False
    p0_latched: bool = False
    p0_codes: tuple[str, ...] = ()
    active_fact_scope: str = ""
    active_topics: tuple[str, ...] = ()
    required_fact_keys: tuple[str, ...] = ()
    turns_seen: int = 0

    def tagger_context(self) -> Mapping[str, object]:
        return {
            "last_transfer_sense": self.transfer_sense,
            "group_topic_active": self.group_topic_active,
        }

    def retrieval_context(self) -> Mapping[str, object]:
        return {
            "active_fact_scope": self.active_fact_scope,
            "active_topics": list(self.active_topics),
            "required_fact_keys": list(self.required_fact_keys),
        }

    def to_json_dict(self) -> Mapping[str, Any]:
        return {
            "schema_version": HELD_STATE_SCHEMA_VERSION,
            "training_format": self.training_format,
            "payment_source": self.payment_source,
            "transfer_sense": self.transfer_sense,
            "group_topic_active": self.group_topic_active,
            "p0_latched": self.p0_latched,
            "p0_codes": list(self.p0_codes),
            "active_fact_scope": self.active_fact_scope,
            "active_topics": list(self.active_topics),
            "required_fact_keys": list(self.required_fact_keys),
            "turns_seen": self.turns_seen,
        }

    def to_prompt_view(self) -> Mapping[str, Any]:
        return self.to_json_dict()


def held_state_from_mapping(payload: Mapping[str, Any] | HeldState | None) -> HeldState:
    if isinstance(payload, HeldState):
        return payload
    data = dict(payload or {})
    return HeldState(
        training_format=str(data.get("training_format") or ""),
        payment_source=str(data.get("payment_source") or ""),
        transfer_sense=str(data.get("transfer_sense") or ""),
        group_topic_active=bool(data.get("group_topic_active")),
        p0_latched=bool(data.get("p0_latched")),
        p0_codes=tuple(str(item) for item in (data.get("p0_codes") or ()) if str(item).strip()),
        active_fact_scope=str(data.get("active_fact_scope") or ""),
        active_topics=tuple(str(item) for item in (data.get("active_topics") or ()) if str(item).strip()),
        required_fact_keys=tuple(str(item) for item in (data.get("required_fact_keys") or ()) if str(item).strip()),
        turns_seen=int(data.get("turns_seen") or 0),
    )


def update_held(
    held: HeldState,
    text: str,
    roles: MessageRoles,
    *,
    p0_required: bool,
    fact_scope: str = "",
    required_fact_keys: tuple[str, ...] = (),
) -> HeldState:
    value = str(text or "")
    new_format = roles.training_format or held.training_format
    new_source = roles.payment_source or held.payment_source
    new_transfer = roles.transfer_sense or held.transfer_sense
    new_group_active = (
        held.group_topic_active
        or roles.transfer_sense == "group"
        or has_any_marker(value, GROUP_TOPIC_CUES)
    )
    new_p0 = held.p0_latched or bool(p0_required)
    codes = held.p0_codes
    if p0_required and roles.refund_frame == "dispute" and "refund" not in codes:
        codes = (*codes, "refund")
    current_topics = tuple(roles.topics)
    current_keys = tuple(required_fact_keys or ())
    return HeldState(
        training_format=new_format,
        payment_source=new_source,
        transfer_sense=new_transfer,
        group_topic_active=new_group_active,
        p0_latched=new_p0,
        p0_codes=codes,
        active_fact_scope=str(fact_scope or "") or held.active_fact_scope,
        active_topics=current_topics or held.active_topics,
        required_fact_keys=current_keys or held.required_fact_keys,
        turns_seen=held.turns_seen + 1,
    )
