from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional, Sequence

from mango_mvp.amocrm_runtime.agent_runtime import ActionProposal
from mango_mvp.channels.contracts import ChannelMessage, ChannelSession, RecommendedAction, normalize_key, stable_digest


CHANNEL_ACTIONS_SCHEMA_VERSION = "channel_actions_v1"
CHANNEL_ACTION_LEVELS = ("L1", "L2", "L3", "L4")

ACTION_DRAFT_CLIENT_MESSAGE = "draft_client_message"
ACTION_REQUEST_CRM_CONTEXT = "request_crm_context"
ACTION_HANDOFF_TO_MANAGER = "handoff_to_manager"
ACTION_CREATE_FOLLOW_UP_TASK = "create_follow_up_task"
ACTION_MARK_MANUAL_REVIEW = "mark_manual_review"
ACTION_NOTIFY_ROP_HOT_LEAD = "notify_rop_hot_lead"

COMMERCIAL_REVIEW_MARKERS = (
    "цена",
    "стоимост",
    "оплат",
    "скидк",
    "рассроч",
    "возврат",
    "договор",
    "гарант",
    "дедлайн",
    "срок",
)
MANAGER_HANDOFF_MARKERS = (
    "менеджер",
    "оператор",
    "человек",
    "позвоните",
    "перезвоните",
    "свяжитесь",
    "связаться",
)
FOLLOW_UP_MARKERS = (
    "перезвон",
    "позвон",
    "свяж",
    "завтра",
    "позже",
    "вечером",
    "утром",
    "после",
)
HOT_LEAD_MARKERS = (
    "срочно",
    "готов оплатить",
    "готова оплатить",
    "готовы оплатить",
    "готов записаться",
    "готова записаться",
    "готовы записаться",
    "оплатить",
    "купить",
    "записаться",
    "бронь",
)


@dataclass(frozen=True)
class ChannelActionPolicy:
    action_type: str
    autonomy_level: str
    requires_approval: bool
    requires_notification: bool = False
    live_execution_allowed: bool = False
    description: str = ""

    def __post_init__(self) -> None:
        action_type = normalize_key(self.action_type, "action_type")
        level = str(self.autonomy_level or "").strip().upper()
        if level not in CHANNEL_ACTION_LEVELS:
            raise ValueError(f"unsupported autonomy level: {self.autonomy_level!r}")
        object.__setattr__(self, "action_type", action_type)
        object.__setattr__(self, "autonomy_level", level)
        if self.live_execution_allowed:
            raise ValueError("channel action policies in this layer must not allow live execution")

    def to_json_dict(self) -> Mapping[str, Any]:
        return asdict(self)


DEFAULT_CHANNEL_ACTION_POLICIES: tuple[ChannelActionPolicy, ...] = (
    ChannelActionPolicy(
        action_type=ACTION_REQUEST_CRM_CONTEXT,
        autonomy_level="L1",
        requires_approval=False,
        description="Запросить read-only CRM/Tallanto context для треда.",
    ),
    ChannelActionPolicy(
        action_type=ACTION_CREATE_FOLLOW_UP_TASK,
        autonomy_level="L2",
        requires_approval=False,
        requires_notification=True,
        description="Предложить follow-up задачу менеджеру без live CRM-записи.",
    ),
    ChannelActionPolicy(
        action_type=ACTION_NOTIFY_ROP_HOT_LEAD,
        autonomy_level="L2",
        requires_approval=False,
        requires_notification=True,
        description="Подсветить РОПу горячий или срочный входящий сигнал.",
    ),
    ChannelActionPolicy(
        action_type=ACTION_DRAFT_CLIENT_MESSAGE,
        autonomy_level="L3",
        requires_approval=True,
        description="Подготовить черновик сообщения клиенту; отправка только после подтверждения.",
    ),
    ChannelActionPolicy(
        action_type=ACTION_HANDOFF_TO_MANAGER,
        autonomy_level="L3",
        requires_approval=True,
        requires_notification=True,
        description="Передать диалог менеджеру, если клиент просит человека или нужен ручной ответ.",
    ),
    ChannelActionPolicy(
        action_type=ACTION_MARK_MANUAL_REVIEW,
        autonomy_level="L3",
        requires_approval=True,
        description="Пометить тред на ручную проверку из-за коммерческого или safety-сигнала.",
    ),
)


def channel_action_policy(action_type: str) -> ChannelActionPolicy:
    normalized = normalize_key(action_type, "action_type")
    for policy in DEFAULT_CHANNEL_ACTION_POLICIES:
        if policy.action_type == normalized:
            return policy
    return ChannelActionPolicy(
        action_type=normalized,
        autonomy_level="L3",
        requires_approval=True,
        description="Безопасная политика по умолчанию: только через ручную проверку.",
    )


def default_channel_action_policy_map() -> Mapping[str, Mapping[str, Any]]:
    return {policy.action_type: policy.to_json_dict() for policy in DEFAULT_CHANNEL_ACTION_POLICIES}


def build_channel_recommended_actions(
    *,
    message: ChannelMessage,
    session: ChannelSession,
    draft_id: str,
    draft_text: str,
    context: Optional[Mapping[str, Any]] = None,
) -> tuple[RecommendedAction, ...]:
    context_payload = dict(context or {})
    actions: list[RecommendedAction] = [
        make_recommended_action(
            action_type=ACTION_DRAFT_CLIENT_MESSAGE,
            target_system="channel",
            entity_type="channel_thread",
            entity_id=session.session_key,
            title="Подготовить черновик ответа",
            summary="Входящее сообщение ожидает проверки менеджером.",
            payload=base_action_payload(
                message=message,
                session=session,
                draft_id=draft_id,
                draft_text=draft_text,
                context=context_payload,
                reason="inbound_message",
            ),
            confidence=0.35,
        ),
        make_recommended_action(
            action_type=ACTION_REQUEST_CRM_CONTEXT,
            target_system="crm_context",
            entity_type="channel_thread",
            entity_id=session.session_key,
            title="Запросить CRM-контекст",
            summary="Перед ответом полезно сопоставить тред с AMO/Tallanto и историей клиента.",
            payload=base_action_payload(
                message=message,
                session=session,
                draft_id=draft_id,
                draft_text=draft_text,
                context=context_payload,
                reason="context_required",
            ),
            confidence=0.72,
        ),
    ]

    lowered = message.text.casefold()
    if has_any_marker(lowered, MANAGER_HANDOFF_MARKERS) or bool(context_payload.get("force_manager_handoff")):
        actions.append(
            make_recommended_action(
                action_type=ACTION_HANDOFF_TO_MANAGER,
                target_system="internal",
                entity_type="channel_thread",
                entity_id=session.session_key,
                title="Передать менеджеру",
                summary="Клиент просит живого человека или нужна ручная коммуникация.",
                payload=base_action_payload(
                    message=message,
                    session=session,
                    draft_id=draft_id,
                    draft_text=draft_text,
                    context=context_payload,
                    reason="manager_handoff_signal",
                ),
                confidence=0.76,
            )
        )

    if has_any_marker(lowered, FOLLOW_UP_MARKERS) or bool(context_payload.get("follow_up_due_at")):
        actions.append(
            make_recommended_action(
                action_type=ACTION_CREATE_FOLLOW_UP_TASK,
                target_system="task_queue",
                entity_type="channel_thread",
                entity_id=session.session_key,
                title="Подготовить follow-up задачу",
                summary="Во входящем сообщении есть сигнал для следующего касания.",
                payload=base_action_payload(
                    message=message,
                    session=session,
                    draft_id=draft_id,
                    draft_text=draft_text,
                    context=context_payload,
                    reason="follow_up_signal",
                )
                | {"due_at": optional_context_text(context_payload, "follow_up_due_at")},
                confidence=0.68,
            )
        )

    if has_any_marker(lowered, COMMERCIAL_REVIEW_MARKERS) or bool(context_payload.get("requires_commercial_review")):
        actions.append(
            make_recommended_action(
                action_type=ACTION_MARK_MANUAL_REVIEW,
                target_system="internal",
                entity_type="channel_thread",
                entity_id=session.session_key,
                title="Проверить коммерческий ответ вручную",
                summary="Сообщение содержит цену, оплату, скидку, рассрочку, договор или похожий риск.",
                payload=base_action_payload(
                    message=message,
                    session=session,
                    draft_id=draft_id,
                    draft_text=draft_text,
                    context=context_payload,
                    reason="commercial_or_policy_signal",
                ),
                confidence=0.74,
            )
        )

    if is_hot_lead_signal(lowered, context_payload):
        actions.append(
            make_recommended_action(
                action_type=ACTION_NOTIFY_ROP_HOT_LEAD,
                target_system="management_queue",
                entity_type="channel_thread",
                entity_id=session.session_key,
                title="Подсветить РОПу горячий сигнал",
                summary="Сообщение выглядит срочным или близким к оплате/записи.",
                payload=base_action_payload(
                    message=message,
                    session=session,
                    draft_id=draft_id,
                    draft_text=draft_text,
                    context=context_payload,
                    reason="hot_lead_signal",
                ),
                confidence=0.7,
            )
        )

    return dedupe_recommended_actions(actions)


def make_recommended_action(
    *,
    action_type: str,
    target_system: str,
    entity_type: str,
    entity_id: Optional[str],
    title: str,
    summary: str,
    payload: Mapping[str, Any],
    confidence: Optional[float],
) -> RecommendedAction:
    policy = channel_action_policy(action_type)
    payload_dict = dict(payload)
    payload_dict["policy"] = {
        "autonomy_level": policy.autonomy_level,
        "requires_approval": policy.requires_approval,
        "requires_notification": policy.requires_notification,
        "live_execution_allowed": policy.live_execution_allowed,
    }
    return RecommendedAction(
        action_type=policy.action_type,
        target_system=target_system,
        entity_type=entity_type,
        entity_id=entity_id,
        title=title,
        summary=summary,
        payload=payload_dict,
        confidence=confidence,
        requires_approval=policy.requires_approval,
        idempotency_key=stable_channel_action_idempotency_key(policy.action_type, payload_dict),
    )


def recommended_action_to_agent_proposal(action: RecommendedAction) -> ActionProposal:
    policy = channel_action_policy(action.action_type)
    return ActionProposal(
        action_type=action.action_type,
        target_system=action.target_system,
        entity_type=action.entity_type,
        entity_id=action.entity_id,
        title=action.title,
        summary=action.summary,
        rationale=f"Channel recommended action. Autonomy={policy.autonomy_level}; live execution disabled in channel layer.",
        payload=dict(action.payload)
        | {
            "channel_action_schema_version": CHANNEL_ACTIONS_SCHEMA_VERSION,
            "channel_action_policy": policy.to_json_dict(),
        },
        confidence=action.confidence,
        idempotency_key=action.idempotency_key,
    )


def recommended_actions_to_agent_proposals(actions: Sequence[RecommendedAction]) -> tuple[ActionProposal, ...]:
    return tuple(recommended_action_to_agent_proposal(action) for action in actions)


def base_action_payload(
    *,
    message: ChannelMessage,
    session: ChannelSession,
    draft_id: str,
    draft_text: str,
    context: Mapping[str, Any],
    reason: str,
) -> Mapping[str, Any]:
    return {
        "schema_version": CHANNEL_ACTIONS_SCHEMA_VERSION,
        "draft_id": draft_id,
        "source_message_idempotency_key": message.idempotency_key,
        "channel": message.channel,
        "channel_thread_id": message.channel_thread_id,
        "channel_user_id": message.channel_user_id,
        "session_key": session.session_key,
        "draft_text": draft_text,
        "reason": reason,
        "requires_approval": True,
        "live_send_enabled": False,
        "context_keys": tuple(sorted(str(key) for key in context.keys())),
    }


def stable_channel_action_idempotency_key(action_type: str, payload: Mapping[str, Any]) -> str:
    return f"recommended_action:{normalize_key(action_type, 'action_type')}:{stable_digest(payload)[:32]}"


def dedupe_recommended_actions(actions: Sequence[RecommendedAction]) -> tuple[RecommendedAction, ...]:
    result: list[RecommendedAction] = []
    seen: set[str] = set()
    for action in actions:
        if not isinstance(action, RecommendedAction):
            raise TypeError("actions must contain RecommendedAction items")
        if action.idempotency_key in seen:
            continue
        seen.add(action.idempotency_key)
        result.append(action)
    return tuple(result)


def has_any_marker(text: str, markers: Sequence[str]) -> bool:
    return any(marker in text for marker in markers)


def is_hot_lead_signal(text: str, context: Mapping[str, Any]) -> bool:
    priority = str(context.get("priority") or context.get("lead_priority") or "").strip().casefold()
    if priority in {"hot", "p0", "high", "горячий", "срочно"}:
        return True
    return has_any_marker(text, HOT_LEAD_MARKERS)


def optional_context_text(context: Mapping[str, Any], key: str) -> Optional[str]:
    value = context.get(key)
    if value is None:
        return None
    text = str(value).strip()
    return text or None
