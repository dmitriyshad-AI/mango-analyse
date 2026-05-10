from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Optional, Sequence

from mango_mvp.channels.actions import (
    ACTION_CREATE_FOLLOW_UP_TASK,
    ACTION_HANDOFF_TO_MANAGER,
    ACTION_MARK_MANUAL_REVIEW,
    ACTION_NOTIFY_ROP_HOT_LEAD,
    ACTION_REQUEST_CRM_CONTEXT,
)
from mango_mvp.channels.contracts import (
    ChannelMessage,
    ChannelSession,
    RecommendedAction,
    normalize_key,
    optional_text,
    require_text,
    require_timezone,
    stable_digest,
)
from mango_mvp.channels.preview_service import ChannelDraftPreview, DEFAULT_DRAFT_TEXT


CHANNEL_SIGNALS_SCHEMA_VERSION = "channel_signals_v1"

SIGNAL_NEED_CRM_CONTEXT = "need_crm_context"
SIGNAL_CUSTOMER_QUESTION = "customer_question"
SIGNAL_URGENCY = "urgency"
SIGNAL_COMMERCIAL_RISK = "commercial_risk"
SIGNAL_MANAGER_HANDOFF = "manager_handoff"
SIGNAL_FOLLOW_UP = "follow_up"
SIGNAL_HOT_LEAD = "hot_lead"
SIGNAL_ATTACHMENT_RECEIVED = "attachment_received"

SIGNAL_ORDER = (
    SIGNAL_NEED_CRM_CONTEXT,
    SIGNAL_CUSTOMER_QUESTION,
    SIGNAL_URGENCY,
    SIGNAL_COMMERCIAL_RISK,
    SIGNAL_MANAGER_HANDOFF,
    SIGNAL_FOLLOW_UP,
    SIGNAL_HOT_LEAD,
    SIGNAL_ATTACHMENT_RECEIVED,
)
SIGNAL_SEVERITIES = {"info", "warning", "high"}
SIGNAL_AUTONOMY_LEVELS = {"L1", "L2", "L3", "L4"}

QUESTION_MARKERS = (
    "?",
    "подскаж",
    "сколько",
    "как ",
    "можно",
    "какой",
    "какая",
    "какие",
    "когда",
    "где",
    "почему",
    "что ",
)
URGENCY_MARKERS = (
    "срочно",
    "сегодня",
    "сейчас",
    "быстро",
    "горит",
    "до вечера",
    "в ближайшее время",
)
COMMERCIAL_MARKERS = (
    "цена",
    "стоимост",
    "оплат",
    "скидк",
    "рассроч",
    "возврат",
    "договор",
    "гарант",
    "счет",
    "счёт",
    "дедлайн",
    "срок",
)
HANDOFF_MARKERS = (
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
    "готов оплатить",
    "готова оплатить",
    "готовы оплатить",
    "готов записаться",
    "готова записаться",
    "готовы записаться",
    "хочу оплатить",
    "хочу купить",
    "хочу записаться",
    "купить",
    "бронь",
)


@dataclass(frozen=True)
class SignalEvidence:
    source_type: str
    source_id: str
    excerpt: str = ""
    markers: Sequence[str] = field(default_factory=tuple)
    weight: float = 1.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_type", normalize_key(self.source_type, "source_type"))
        object.__setattr__(self, "source_id", require_text(self.source_id, "source_id"))
        object.__setattr__(self, "excerpt", trim_excerpt(self.excerpt))
        markers = tuple(str(item).strip() for item in self.markers if str(item).strip())
        object.__setattr__(self, "markers", markers)
        if not 0 <= float(self.weight) <= 1:
            raise ValueError("evidence weight must be between 0 and 1")
        object.__setattr__(self, "weight", float(self.weight))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_json_dict(self) -> Mapping[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SignalPolicy:
    signal_type: str
    autonomy_level: str
    requires_manager_review: bool = True
    requires_notification: bool = False
    allow_autonomous_reply: bool = False
    allow_live_execution: bool = False
    description: str = ""

    def __post_init__(self) -> None:
        signal_type = normalize_key(self.signal_type, "signal_type")
        level = str(self.autonomy_level or "").strip().upper()
        if level not in SIGNAL_AUTONOMY_LEVELS:
            raise ValueError(f"unsupported signal autonomy level: {self.autonomy_level!r}")
        if self.allow_autonomous_reply:
            raise ValueError("channel signal policies must not allow autonomous client replies")
        if self.allow_live_execution:
            raise ValueError("channel signal policies must not allow live execution")
        object.__setattr__(self, "signal_type", signal_type)
        object.__setattr__(self, "autonomy_level", level)

    def to_json_dict(self) -> Mapping[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CustomerSignal:
    signal_type: str
    title: str
    summary: str
    evidence: Sequence[SignalEvidence] = field(default_factory=tuple)
    confidence: float = 0.5
    severity: str = "info"
    policy: Optional[SignalPolicy] = None
    source_action_types: Sequence[str] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    idempotency_key: Optional[str] = None

    def __post_init__(self) -> None:
        signal_type = normalize_key(self.signal_type, "signal_type")
        severity = normalize_key(self.severity, "severity")
        if severity not in SIGNAL_SEVERITIES:
            raise ValueError(f"unsupported signal severity: {self.severity!r}")
        evidence = tuple(self.evidence)
        if any(not isinstance(item, SignalEvidence) for item in evidence):
            raise TypeError("evidence must contain SignalEvidence items")
        confidence = float(self.confidence)
        if not 0 <= confidence <= 1:
            raise ValueError("signal confidence must be between 0 and 1")
        source_action_types = tuple(normalize_key(item, "source_action_type") for item in self.source_action_types)
        policy = self.policy or signal_policy(signal_type)
        key = optional_text(self.idempotency_key) or stable_signal_idempotency_key(
            signal_type=signal_type,
            evidence=evidence,
            source_action_types=source_action_types,
            metadata=self.metadata,
        )
        object.__setattr__(self, "signal_type", signal_type)
        object.__setattr__(self, "title", require_text(self.title, "signal title"))
        object.__setattr__(self, "summary", require_text(self.summary, "signal summary"))
        object.__setattr__(self, "evidence", evidence)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "severity", severity)
        object.__setattr__(self, "policy", policy)
        object.__setattr__(self, "source_action_types", source_action_types)
        object.__setattr__(self, "metadata", dict(self.metadata))
        object.__setattr__(self, "idempotency_key", key)

    @property
    def requires_manager_review(self) -> bool:
        return bool(self.policy.requires_manager_review if self.policy else True)

    def to_json_dict(self) -> Mapping[str, Any]:
        return {
            "schema_version": CHANNEL_SIGNALS_SCHEMA_VERSION,
            "signal_type": self.signal_type,
            "title": self.title,
            "summary": self.summary,
            "evidence": [item.to_json_dict() for item in self.evidence],
            "confidence": self.confidence,
            "severity": self.severity,
            "policy": self.policy.to_json_dict() if self.policy else None,
            "source_action_types": list(self.source_action_types),
            "metadata": dict(self.metadata),
            "idempotency_key": self.idempotency_key,
        }


@dataclass(frozen=True)
class SafeAnswer:
    text: str
    answer_type: str = "draft"
    requires_approval: bool = True
    blocked_reasons: Sequence[str] = field(default_factory=tuple)
    source_signal_types: Sequence[str] = field(default_factory=tuple)
    safety_flags: Sequence[str] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        answer_type = normalize_key(self.answer_type, "answer_type")
        if answer_type != "draft":
            raise ValueError("SafeAnswer supports draft answers only in channel signal layer")
        if not self.requires_approval:
            raise ValueError("SafeAnswer must require approval in channel signal layer")
        blocked = tuple(normalize_key(item, "blocked_reason") for item in self.blocked_reasons)
        signal_types = tuple(normalize_key(item, "source_signal_type") for item in self.source_signal_types)
        safety = tuple(normalize_key(item, "safety_flag") for item in self.safety_flags)
        object.__setattr__(self, "text", require_text(self.text, "safe answer text"))
        object.__setattr__(self, "answer_type", answer_type)
        object.__setattr__(self, "blocked_reasons", blocked)
        object.__setattr__(self, "source_signal_types", signal_types)
        object.__setattr__(self, "safety_flags", safety)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_json_dict(self) -> Mapping[str, Any]:
        return {
            "schema_version": CHANNEL_SIGNALS_SCHEMA_VERSION,
            "text": self.text,
            "answer_type": self.answer_type,
            "requires_approval": self.requires_approval,
            "blocked_reasons": list(self.blocked_reasons),
            "source_signal_types": list(self.source_signal_types),
            "safety_flags": list(self.safety_flags),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class SignalDecision:
    decision_id: str
    message_idempotency_key: str
    session_key: str
    signals: Sequence[CustomerSignal]
    safe_answer: SafeAnswer
    recommended_action_types: Sequence[str] = field(default_factory=tuple)
    policy_flags: Mapping[str, bool] = field(default_factory=dict)
    blocked_reasons: Sequence[str] = field(default_factory=tuple)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "decision_id", require_text(self.decision_id, "decision_id"))
        object.__setattr__(
            self,
            "message_idempotency_key",
            require_text(self.message_idempotency_key, "message_idempotency_key"),
        )
        object.__setattr__(self, "session_key", require_text(self.session_key, "session_key"))
        signals = tuple(self.signals)
        if any(not isinstance(item, CustomerSignal) for item in signals):
            raise TypeError("signals must contain CustomerSignal items")
        if not isinstance(self.safe_answer, SafeAnswer):
            raise TypeError("safe_answer must be SafeAnswer")
        action_types = tuple(normalize_key(item, "recommended_action_type") for item in self.recommended_action_types)
        blocked = tuple(normalize_key(item, "blocked_reason") for item in self.blocked_reasons)
        require_timezone(self.created_at, "created_at")
        object.__setattr__(self, "signals", signals)
        object.__setattr__(self, "recommended_action_types", action_types)
        object.__setattr__(self, "policy_flags", dict(self.policy_flags))
        object.__setattr__(self, "blocked_reasons", blocked)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def requires_manager_review(self) -> bool:
        return bool(self.policy_flags.get("requires_manager_review") or self.safe_answer.requires_approval)

    def to_json_dict(self) -> Mapping[str, Any]:
        return {
            "schema_version": CHANNEL_SIGNALS_SCHEMA_VERSION,
            "decision_id": self.decision_id,
            "message_idempotency_key": self.message_idempotency_key,
            "session_key": self.session_key,
            "signals": [item.to_json_dict() for item in self.signals],
            "safe_answer": self.safe_answer.to_json_dict(),
            "recommended_action_types": list(self.recommended_action_types),
            "policy_flags": dict(self.policy_flags),
            "blocked_reasons": list(self.blocked_reasons),
            "created_at": self.created_at.isoformat(),
            "metadata": dict(self.metadata),
        }


DEFAULT_SIGNAL_POLICIES = (
    SignalPolicy(
        signal_type=SIGNAL_NEED_CRM_CONTEXT,
        autonomy_level="L1",
        requires_manager_review=False,
        description="Можно запрашивать read-only context, но без CRM-записи.",
    ),
    SignalPolicy(
        signal_type=SIGNAL_CUSTOMER_QUESTION,
        autonomy_level="L3",
        requires_manager_review=True,
        description="Ответ клиенту только как черновик для менеджера.",
    ),
    SignalPolicy(
        signal_type=SIGNAL_URGENCY,
        autonomy_level="L2",
        requires_manager_review=False,
        requires_notification=True,
        description="Срочный сигнал можно подсветить, но не отвечать автономно.",
    ),
    SignalPolicy(
        signal_type=SIGNAL_COMMERCIAL_RISK,
        autonomy_level="L3",
        requires_manager_review=True,
        description="Коммерческие условия требуют ручной проверки.",
    ),
    SignalPolicy(
        signal_type=SIGNAL_MANAGER_HANDOFF,
        autonomy_level="L3",
        requires_manager_review=True,
        requires_notification=True,
        description="Передача человеку требует контролируемого workflow.",
    ),
    SignalPolicy(
        signal_type=SIGNAL_FOLLOW_UP,
        autonomy_level="L2",
        requires_manager_review=False,
        requires_notification=True,
        description="Можно предложить follow-up без live CRM-записи.",
    ),
    SignalPolicy(
        signal_type=SIGNAL_HOT_LEAD,
        autonomy_level="L2",
        requires_manager_review=False,
        requires_notification=True,
        description="Можно подсветить РОПу горячий лид без client-facing auto-send.",
    ),
    SignalPolicy(
        signal_type=SIGNAL_ATTACHMENT_RECEIVED,
        autonomy_level="L3",
        requires_manager_review=True,
        description="Вложения требуют ручной проверки перед ответом.",
    ),
)


def signal_policy(signal_type: str) -> SignalPolicy:
    normalized = normalize_key(signal_type, "signal_type")
    for policy in DEFAULT_SIGNAL_POLICIES:
        if policy.signal_type == normalized:
            return policy
    return SignalPolicy(
        signal_type=normalized,
        autonomy_level="L3",
        requires_manager_review=True,
        description="Безопасная политика по умолчанию: только через ручную проверку.",
    )


def default_signal_policy_map() -> Mapping[str, Mapping[str, Any]]:
    return {policy.signal_type: policy.to_json_dict() for policy in DEFAULT_SIGNAL_POLICIES}


def build_channel_signal_decision(
    *,
    message: ChannelMessage,
    session: Optional[ChannelSession] = None,
    preview: Optional[ChannelDraftPreview] = None,
    actions: Optional[Sequence[RecommendedAction]] = None,
    context: Optional[Mapping[str, Any]] = None,
) -> SignalDecision:
    resolved_session = session or (preview.session if preview else ChannelSession.from_message(message))
    if resolved_session.channel != message.channel:
        raise ValueError("session channel must match source message channel")
    if resolved_session.channel_thread_id != message.channel_thread_id:
        raise ValueError("session thread must match source message thread")
    action_items = tuple(actions if actions is not None else (preview.reply.recommended_actions if preview else ()))
    signals = extract_customer_signals(
        message=message,
        session=resolved_session,
        actions=action_items,
        context=context,
    )
    action_types = tuple(action.action_type for action in action_items)
    safe_answer = build_safe_answer(
        message=message,
        preview=preview,
        signals=signals,
        context=context,
    )
    flags = build_policy_flags(signals=signals, safe_answer=safe_answer)
    blocked_reasons = build_decision_blocked_reasons(signals, safe_answer)
    decision_id = stable_signal_decision_id(
        message=message,
        session=resolved_session,
        signals=signals,
        action_types=action_types,
    )
    return SignalDecision(
        decision_id=decision_id,
        message_idempotency_key=message.idempotency_key,
        session_key=resolved_session.session_key,
        signals=signals,
        safe_answer=safe_answer,
        recommended_action_types=action_types,
        policy_flags=flags,
        blocked_reasons=blocked_reasons,
        metadata={
            "signal_count": len(signals),
            "context_keys": tuple(sorted(str(key) for key in (context or {}).keys())),
            "preview_present": preview is not None,
        },
    )


def extract_customer_signals(
    *,
    message: ChannelMessage,
    session: ChannelSession,
    actions: Sequence[RecommendedAction] = (),
    context: Optional[Mapping[str, Any]] = None,
) -> tuple[CustomerSignal, ...]:
    if session.channel != message.channel:
        raise ValueError("session channel must match source message channel")
    if session.channel_thread_id != message.channel_thread_id:
        raise ValueError("session thread must match source message thread")
    context_payload = dict(context or {})
    lowered = message.text.casefold()
    by_type: dict[str, list[SignalEvidence]] = {signal_type: [] for signal_type in SIGNAL_ORDER}
    action_types = {action.action_type for action in actions}

    add_marker_evidence(
        by_type,
        SIGNAL_CUSTOMER_QUESTION,
        message=message,
        markers=find_markers(lowered, QUESTION_MARKERS),
        weight=0.72,
    )
    add_marker_evidence(
        by_type,
        SIGNAL_URGENCY,
        message=message,
        markers=find_markers(lowered, URGENCY_MARKERS),
        weight=0.7,
    )
    add_marker_evidence(
        by_type,
        SIGNAL_COMMERCIAL_RISK,
        message=message,
        markers=find_markers(lowered, COMMERCIAL_MARKERS),
        weight=0.76,
    )
    add_marker_evidence(
        by_type,
        SIGNAL_MANAGER_HANDOFF,
        message=message,
        markers=find_markers(lowered, HANDOFF_MARKERS),
        weight=0.74,
    )
    add_marker_evidence(
        by_type,
        SIGNAL_FOLLOW_UP,
        message=message,
        markers=find_markers(lowered, FOLLOW_UP_MARKERS),
        weight=0.7,
    )
    add_marker_evidence(
        by_type,
        SIGNAL_HOT_LEAD,
        message=message,
        markers=find_markers(lowered, HOT_LEAD_MARKERS),
        weight=0.75,
    )
    if message.attachments:
        by_type[SIGNAL_ATTACHMENT_RECEIVED].append(
            SignalEvidence(
                source_type="channel_message",
                source_id=message.idempotency_key,
                excerpt=f"{len(message.attachments)} attachment(s)",
                markers=tuple(item.kind for item in message.attachments),
                weight=0.85,
                metadata={"attachment_count": len(message.attachments)},
            )
        )

    action_signal_map = {
        ACTION_REQUEST_CRM_CONTEXT: SIGNAL_NEED_CRM_CONTEXT,
        ACTION_HANDOFF_TO_MANAGER: SIGNAL_MANAGER_HANDOFF,
        ACTION_CREATE_FOLLOW_UP_TASK: SIGNAL_FOLLOW_UP,
        ACTION_MARK_MANUAL_REVIEW: SIGNAL_COMMERCIAL_RISK,
        ACTION_NOTIFY_ROP_HOT_LEAD: SIGNAL_HOT_LEAD,
    }
    for action in actions:
        signal_type = action_signal_map.get(action.action_type)
        if not signal_type:
            continue
        by_type[signal_type].append(
            SignalEvidence(
                source_type="recommended_action",
                source_id=action.idempotency_key or action.action_type,
                excerpt=action.summary or action.title,
                markers=(action.action_type,),
                weight=float(action.confidence if action.confidence is not None else 0.65),
                metadata={
                    "action_type": action.action_type,
                    "target_system": action.target_system,
                    "requires_approval": action.requires_approval,
                },
            )
        )

    context_signal_pairs = (
        (SIGNAL_NEED_CRM_CONTEXT, ("crm_context_missing", "needs_crm_context", "request_crm_context")),
        (SIGNAL_URGENCY, ("urgent", "sla_overdue")),
        (SIGNAL_COMMERCIAL_RISK, ("requires_commercial_review", "commercial_risk")),
        (SIGNAL_MANAGER_HANDOFF, ("force_manager_handoff", "handoff_required")),
        (SIGNAL_FOLLOW_UP, ("follow_up_due_at", "follow_up_required")),
        (SIGNAL_HOT_LEAD, ("priority", "lead_priority", "hot_lead")),
    )
    for signal_type, keys in context_signal_pairs:
        matched_keys = matched_context_keys(signal_type, context_payload, keys)
        if matched_keys:
            by_type[signal_type].append(
                SignalEvidence(
                    source_type="read_only_context",
                    source_id=context_source_id(message, matched_keys),
                    excerpt=", ".join(f"{key}={context_payload.get(key)!r}" for key in matched_keys),
                    markers=matched_keys,
                    weight=0.7,
                    metadata={"context_keys": matched_keys},
                )
            )

    signals: list[CustomerSignal] = []
    for signal_type in SIGNAL_ORDER:
        evidence = tuple(by_type[signal_type])
        if not evidence:
            continue
        source_action_types = tuple(sorted(action_types_for_signal(signal_type, action_types)))
        signals.append(
            CustomerSignal(
                signal_type=signal_type,
                title=signal_title(signal_type),
                summary=signal_summary(signal_type),
                evidence=evidence,
                confidence=signal_confidence(evidence),
                severity=signal_severity(signal_type, evidence),
                policy=signal_policy(signal_type),
                source_action_types=source_action_types,
                metadata={
                    "channel": message.channel,
                    "session_key": session.session_key,
                    "evidence_count": len(evidence),
                },
            )
        )
    return tuple(signals)


def build_safe_answer(
    *,
    message: ChannelMessage,
    preview: Optional[ChannelDraftPreview],
    signals: Sequence[CustomerSignal],
    context: Optional[Mapping[str, Any]] = None,
) -> SafeAnswer:
    context_payload = dict(context or {})
    text = optional_text(context_payload.get("safe_draft_text")) or (preview.reply.text if preview else None) or DEFAULT_DRAFT_TEXT
    signal_types = tuple(signal.signal_type for signal in signals)
    blocked_reasons = ["client_facing_auto_send_disabled", "manager_approval_required", "no_validated_answer_base"]
    if any(signal.signal_type == SIGNAL_COMMERCIAL_RISK for signal in signals):
        blocked_reasons.append("commercial_review_required")
    if message.attachments:
        blocked_reasons.append("attachment_review_required")
    return SafeAnswer(
        text=text,
        answer_type="draft",
        requires_approval=True,
        blocked_reasons=dedupe_texts(blocked_reasons),
        source_signal_types=signal_types,
        safety_flags=(
            "draft_only",
            "requires_manager_approval",
            "live_send_disabled",
            "no_llm_used",
            "no_rag_used",
            "no_crm_write",
        ),
        metadata={
            "source": "channel_signal_engine",
            "preview_present": preview is not None,
            "channel": message.channel,
        },
    )


def build_policy_flags(*, signals: Sequence[CustomerSignal], safe_answer: SafeAnswer) -> Mapping[str, bool]:
    return {
        "requires_manager_review": bool(safe_answer.requires_approval or any(signal.requires_manager_review for signal in signals)),
        "requires_crm_context": any(signal.signal_type == SIGNAL_NEED_CRM_CONTEXT for signal in signals),
        "requires_notification": any(signal.policy.requires_notification for signal in signals if signal.policy),
        "notify_rop": any(signal.signal_type == SIGNAL_HOT_LEAD for signal in signals),
        "allow_autonomous_reply": False,
        "allow_live_execution": False,
        "network_calls": False,
        "llm_calls": False,
        "rag_used": False,
        "write_crm": False,
        "write_tallanto": False,
        "write_runtime_db": False,
        "run_asr": False,
        "run_ra": False,
    }


def build_decision_blocked_reasons(signals: Sequence[CustomerSignal], safe_answer: SafeAnswer) -> tuple[str, ...]:
    reasons = list(safe_answer.blocked_reasons)
    if any(signal.signal_type == SIGNAL_HOT_LEAD for signal in signals):
        reasons.append("hot_lead_requires_human_or_rop_review")
    if any(signal.signal_type == SIGNAL_MANAGER_HANDOFF for signal in signals):
        reasons.append("manager_handoff_required")
    return dedupe_texts(reasons)


def add_marker_evidence(
    by_type: dict[str, list[SignalEvidence]],
    signal_type: str,
    *,
    message: ChannelMessage,
    markers: Sequence[str],
    weight: float,
) -> None:
    if not markers:
        return
    by_type[signal_type].append(
        SignalEvidence(
            source_type="channel_message",
            source_id=message.idempotency_key,
            excerpt=message.text,
            markers=markers,
            weight=weight,
            metadata={"channel": message.channel},
        )
    )


def find_markers(text: str, markers: Sequence[str]) -> tuple[str, ...]:
    return tuple(marker for marker in markers if marker in text)


def signal_confidence(evidence: Sequence[SignalEvidence]) -> float:
    if not evidence:
        return 0.0
    strongest = max(item.weight for item in evidence)
    count_bonus = min(0.18, 0.04 * (len(evidence) - 1))
    return min(0.95, round(strongest + count_bonus, 4))


def signal_severity(signal_type: str, evidence: Sequence[SignalEvidence]) -> str:
    if signal_type in {SIGNAL_COMMERCIAL_RISK, SIGNAL_HOT_LEAD}:
        return "high"
    if signal_type in {SIGNAL_URGENCY, SIGNAL_MANAGER_HANDOFF, SIGNAL_FOLLOW_UP, SIGNAL_ATTACHMENT_RECEIVED}:
        return "warning"
    return "info"


def signal_title(signal_type: str) -> str:
    return {
        SIGNAL_NEED_CRM_CONTEXT: "Нужен CRM-контекст",
        SIGNAL_CUSTOMER_QUESTION: "Вопрос клиента",
        SIGNAL_URGENCY: "Срочный сигнал",
        SIGNAL_COMMERCIAL_RISK: "Коммерческий риск",
        SIGNAL_MANAGER_HANDOFF: "Нужен менеджер",
        SIGNAL_FOLLOW_UP: "Нужен follow-up",
        SIGNAL_HOT_LEAD: "Горячий лид",
        SIGNAL_ATTACHMENT_RECEIVED: "Получено вложение",
    }.get(signal_type, "Сигнал клиента")


def signal_summary(signal_type: str) -> str:
    return {
        SIGNAL_NEED_CRM_CONTEXT: "Перед ответом нужно сопоставить обращение с read-only CRM/Tallanto context.",
        SIGNAL_CUSTOMER_QUESTION: "Сообщение похоже на вопрос или запрос информации от клиента.",
        SIGNAL_URGENCY: "Сообщение содержит маркер срочности или короткого SLA.",
        SIGNAL_COMMERCIAL_RISK: "Сообщение касается цены, оплаты, скидки, договора, сроков или похожих условий.",
        SIGNAL_MANAGER_HANDOFF: "Клиент просит человека, звонок или ручную коммуникацию.",
        SIGNAL_FOLLOW_UP: "В сообщении есть сигнал для следующего касания.",
        SIGNAL_HOT_LEAD: "Сообщение похоже на близкий к покупке или записи лид.",
        SIGNAL_ATTACHMENT_RECEIVED: "Сообщение содержит вложение, которое нужно проверить перед ответом.",
    }.get(signal_type, "Канальный сигнал требует безопасной обработки.")


def action_types_for_signal(signal_type: str, action_types: set[str]) -> tuple[str, ...]:
    mapping = {
        SIGNAL_NEED_CRM_CONTEXT: (ACTION_REQUEST_CRM_CONTEXT,),
        SIGNAL_COMMERCIAL_RISK: (ACTION_MARK_MANUAL_REVIEW,),
        SIGNAL_MANAGER_HANDOFF: (ACTION_HANDOFF_TO_MANAGER,),
        SIGNAL_FOLLOW_UP: (ACTION_CREATE_FOLLOW_UP_TASK,),
        SIGNAL_HOT_LEAD: (ACTION_NOTIFY_ROP_HOT_LEAD,),
    }
    return tuple(action_type for action_type in mapping.get(signal_type, ()) if action_type in action_types)


def context_truthy(context: Mapping[str, Any], key: str) -> bool:
    if key not in context:
        return False
    value = context.get(key)
    if isinstance(value, str):
        return bool(value.strip())
    return bool(value)


def matched_context_keys(signal_type: str, context: Mapping[str, Any], keys: Sequence[str]) -> tuple[str, ...]:
    matched: list[str] = []
    for key in keys:
        if signal_type == SIGNAL_HOT_LEAD and key in {"priority", "lead_priority"}:
            priority = str(context.get(key) or "").strip().casefold()
            if priority in {"hot", "p0", "high", "горячий", "срочно"}:
                matched.append(key)
            continue
        if context_truthy(context, key):
            matched.append(key)
    return tuple(matched)


def context_source_id(message: ChannelMessage, keys: Sequence[str]) -> str:
    return f"read_only_context:{message.idempotency_key}:{stable_digest({'keys': list(keys)})[:12]}"


def stable_signal_idempotency_key(
    *,
    signal_type: str,
    evidence: Sequence[SignalEvidence],
    source_action_types: Sequence[str],
    metadata: Mapping[str, Any],
) -> str:
    digest = stable_digest(
        {
            "signal_type": signal_type,
            "evidence": [item.to_json_dict() for item in evidence],
            "source_action_types": list(source_action_types),
            "metadata": dict(metadata),
        }
    )
    return f"customer_signal:{signal_type}:{digest[:32]}"


def stable_signal_decision_id(
    *,
    message: ChannelMessage,
    session: ChannelSession,
    signals: Sequence[CustomerSignal],
    action_types: Sequence[str],
) -> str:
    digest = stable_digest(
        {
            "message": message.idempotency_key,
            "session": session.session_key,
            "signals": [signal.idempotency_key for signal in signals],
            "action_types": list(action_types),
        }
    )
    return f"signal_decision:{digest[:32]}"


def dedupe_texts(values: Sequence[str]) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = normalize_key(value, "text_key")
        if normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return tuple(result)


def trim_excerpt(value: Any, *, limit: int = 180) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "..."


def signal_engine_safety_contract() -> Mapping[str, bool]:
    return {
        "network_calls": False,
        "llm_calls": False,
        "rag_used": False,
        "live_send": False,
        "write_crm": False,
        "write_tallanto": False,
        "write_runtime_db": False,
        "run_asr": False,
        "run_ra": False,
        "allow_autonomous_reply": False,
    }
