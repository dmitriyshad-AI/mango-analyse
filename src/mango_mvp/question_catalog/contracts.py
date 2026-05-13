from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Optional, Sequence


QUESTION_CATALOG_CONTRACTS_SCHEMA_VERSION = "question_catalog_contracts_v1"
QUESTION_CATALOG_SAFETY_SCHEMA_VERSION = "question_catalog_safety_v1"

SOURCE_CALL = "call"
SOURCE_TELEGRAM = "telegram"
SOURCE_EMAIL = "email"

ANSWER_STATUS_APPROVED = "approved_answer_exists"
ANSWER_STATUS_DRAFT_NEEDS_REVIEW = "draft_answer_exists_needs_review"
ANSWER_STATUS_NEEDS_ROP_ANSWER = "needs_rop_answer"
ANSWER_STATUS_MANAGER_ONLY = "manager_only"
ANSWER_STATUS_SOURCE_CONFLICT = "source_conflict"
ANSWER_STATUS_TIME_SENSITIVE = "outdated_or_time_sensitive"
ANSWER_STATUS_TEMPLATE_NEEDS_CURRENT_FACT = "template_ready_needs_current_fact"
ANSWER_STATUS_FACT_MISSING_OR_STALE = "fact_missing_or_stale"
ANSWER_STATUS_NOT_ENOUGH_CONTEXT = "not_enough_context"

BOT_PERMISSION_ALLOWED_AFTER_FACT_CHECK = "allowed_after_fact_check"
BOT_PERMISSION_DRAFT_ONLY = "draft_only_needs_review"
BOT_PERMISSION_MANAGER_ONLY = "manager_only"
BOT_PERMISSION_NOT_ALLOWED = "not_allowed"

FACT_TYPE_PRICE = "price"
FACT_TYPE_SCHEDULE = "schedule"
FACT_TYPE_LOCATION = "location"
FACT_TYPE_DISCOUNT = "discount"
FACT_TYPE_INSTALLMENT = "installment"
FACT_TYPE_TRIAL = "trial"
FACT_TYPE_DOCUMENTS = "documents"
FACT_TYPE_PROGRAM = "program"

_KEY_RE = re.compile(r"^[a-z0-9][a-z0-9_.:-]{0,119}$")
_ALLOWED_SOURCES = {SOURCE_CALL, SOURCE_TELEGRAM, SOURCE_EMAIL}
_ALLOWED_STATUSES = {
    ANSWER_STATUS_APPROVED,
    ANSWER_STATUS_DRAFT_NEEDS_REVIEW,
    ANSWER_STATUS_NEEDS_ROP_ANSWER,
    ANSWER_STATUS_MANAGER_ONLY,
    ANSWER_STATUS_SOURCE_CONFLICT,
    ANSWER_STATUS_TIME_SENSITIVE,
    ANSWER_STATUS_TEMPLATE_NEEDS_CURRENT_FACT,
    ANSWER_STATUS_FACT_MISSING_OR_STALE,
    ANSWER_STATUS_NOT_ENOUGH_CONTEXT,
}
_ALLOWED_BOT_PERMISSIONS = {
    BOT_PERMISSION_ALLOWED_AFTER_FACT_CHECK,
    BOT_PERMISSION_DRAFT_ONLY,
    BOT_PERMISSION_MANAGER_ONLY,
    BOT_PERMISSION_NOT_ALLOWED,
}


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def stable_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def stable_prefixed_id(prefix: str, payload: Mapping[str, Any], *, length: int = 32) -> str:
    return f"{normalize_key(prefix, 'id prefix')}:{stable_digest(payload)[:length]}"


def normalize_key(value: Any, field_name: str) -> str:
    normalized = str(value or "").strip().lower().replace(" ", "_")
    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
    if not _KEY_RE.match(normalized):
        raise ValueError(f"{field_name} contains unsupported characters: {value!r}")
    return normalized


def require_text(value: Any, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field_name} must not be empty")
    return text


def optional_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def normalize_sequence(values: Sequence[Any] | None, field_name: str) -> tuple[str, ...]:
    if not values:
        return ()
    result = tuple(dict.fromkeys(str(item).strip() for item in values if str(item).strip()))
    if any("\n" in item or "\r" in item for item in result):
        raise ValueError(f"{field_name} must not contain newlines")
    return result


def require_timezone(value: datetime, field_name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")


def _dt_to_json(value: Optional[datetime]) -> Optional[str]:
    return value.isoformat() if value else None


def stable_question_item_id(
    *,
    tenant_id: str,
    source_channel: str,
    source_ref: str,
    customer_text_redacted: str,
) -> str:
    return stable_prefixed_id(
        "question_item",
        {
            "tenant_id": normalize_key(tenant_id, "tenant_id"),
            "source_channel": normalize_source_channel(source_channel),
            "source_ref": require_text(source_ref, "source_ref"),
            "customer_text_redacted": require_text(customer_text_redacted, "customer_text_redacted"),
        },
    )


def stable_question_class_id(*, tenant_id: str, class_key: str) -> str:
    return stable_prefixed_id(
        "question_class",
        {
            "tenant_id": normalize_key(tenant_id, "tenant_id"),
            "class_key": require_text(class_key, "class_key"),
        },
    )


def stable_answer_template_id(*, tenant_id: str, question_class_id: str, template_text: str) -> str:
    return stable_prefixed_id(
        "answer_template",
        {
            "tenant_id": normalize_key(tenant_id, "tenant_id"),
            "question_class_id": require_text(question_class_id, "question_class_id"),
            "template_text": require_text(template_text, "template_text"),
        },
    )


def normalize_source_channel(value: Any) -> str:
    source = normalize_key(value, "source_channel")
    if source not in _ALLOWED_SOURCES:
        raise ValueError(f"unsupported source_channel: {value!r}")
    return source


def normalize_answer_status(value: Any) -> str:
    status = normalize_key(value, "answer_status")
    if status not in _ALLOWED_STATUSES:
        raise ValueError(f"unsupported answer_status: {value!r}")
    return status


def normalize_bot_permission(value: Any) -> str:
    permission = normalize_key(value, "bot_permission")
    if permission not in _ALLOWED_BOT_PERMISSIONS:
        raise ValueError(f"unsupported bot_permission: {value!r}")
    return permission


@dataclass(frozen=True)
class QuestionItem:
    tenant_id: str
    source_channel: str
    source_ref: str
    customer_text_redacted: str
    question_class_id: str
    occurred_at: Optional[datetime] = None
    manager_text_redacted: Optional[str] = None
    intent: str = "other"
    product: Optional[str] = None
    grade: Optional[str] = None
    subject: Optional[str] = None
    format: Optional[str] = None
    price_related: bool = False
    schedule_related: bool = False
    documents_related: bool = False
    safety_flags: Sequence[str] = field(default_factory=tuple)
    answer_evidence_status: str = ANSWER_STATUS_NEEDS_ROP_ANSWER
    answer_source: Optional[str] = None
    requires_dynamic_facts: bool = False
    dynamic_fact_types: Sequence[str] = field(default_factory=tuple)
    fact_freshness_required: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    question_item_id: Optional[str] = None

    def __post_init__(self) -> None:
        tenant_id = normalize_key(self.tenant_id, "tenant_id")
        source_channel = normalize_source_channel(self.source_channel)
        source_ref = require_text(self.source_ref, "source_ref")
        customer_text = require_text(self.customer_text_redacted, "customer_text_redacted")
        if self.occurred_at is not None:
            require_timezone(self.occurred_at, "occurred_at")
        status = normalize_answer_status(self.answer_evidence_status)
        item_id = optional_text(self.question_item_id) or stable_question_item_id(
            tenant_id=tenant_id,
            source_channel=source_channel,
            source_ref=source_ref,
            customer_text_redacted=customer_text,
        )
        object.__setattr__(self, "tenant_id", tenant_id)
        object.__setattr__(self, "source_channel", source_channel)
        object.__setattr__(self, "source_ref", source_ref)
        object.__setattr__(self, "customer_text_redacted", customer_text)
        object.__setattr__(self, "question_class_id", require_text(self.question_class_id, "question_class_id"))
        object.__setattr__(self, "manager_text_redacted", optional_text(self.manager_text_redacted))
        object.__setattr__(self, "intent", normalize_key(self.intent, "intent"))
        object.__setattr__(self, "product", optional_text(self.product))
        object.__setattr__(self, "grade", optional_text(self.grade))
        object.__setattr__(self, "subject", optional_text(self.subject))
        object.__setattr__(self, "format", optional_text(self.format))
        object.__setattr__(self, "safety_flags", normalize_sequence(self.safety_flags, "safety_flags"))
        object.__setattr__(self, "answer_evidence_status", status)
        object.__setattr__(self, "answer_source", optional_text(self.answer_source))
        object.__setattr__(self, "dynamic_fact_types", normalize_sequence(self.dynamic_fact_types, "dynamic_fact_types"))
        object.__setattr__(self, "fact_freshness_required", optional_text(self.fact_freshness_required))
        object.__setattr__(self, "metadata", dict(self.metadata))
        object.__setattr__(self, "question_item_id", item_id)

    def to_json_dict(self) -> Mapping[str, Any]:
        return {
            "schema_version": QUESTION_CATALOG_CONTRACTS_SCHEMA_VERSION,
            "question_item_id": self.question_item_id,
            "tenant_id": self.tenant_id,
            "source_channel": self.source_channel,
            "source_ref": self.source_ref,
            "occurred_at": _dt_to_json(self.occurred_at),
            "customer_text_redacted": self.customer_text_redacted,
            "manager_text_redacted": self.manager_text_redacted,
            "question_class_id": self.question_class_id,
            "intent": self.intent,
            "product": self.product,
            "grade": self.grade,
            "subject": self.subject,
            "format": self.format,
            "price_related": self.price_related,
            "schedule_related": self.schedule_related,
            "documents_related": self.documents_related,
            "safety_flags": list(self.safety_flags),
            "answer_evidence_status": self.answer_evidence_status,
            "answer_source": self.answer_source,
            "requires_dynamic_facts": self.requires_dynamic_facts,
            "dynamic_fact_types": list(self.dynamic_fact_types),
            "fact_freshness_required": self.fact_freshness_required,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class QuestionClass:
    tenant_id: str
    canonical_question: str
    narrow_scope: str
    class_key: str
    exclusions: str = ""
    examples_redacted: Sequence[str] = field(default_factory=tuple)
    count_total: int = 0
    count_calls: int = 0
    count_telegram: int = 0
    count_email: int = 0
    first_seen_at: Optional[datetime] = None
    last_seen_at: Optional[datetime] = None
    products: Sequence[str] = field(default_factory=tuple)
    grades: Sequence[str] = field(default_factory=tuple)
    subjects: Sequence[str] = field(default_factory=tuple)
    answer_status: str = ANSWER_STATUS_NEEDS_ROP_ANSWER
    answer_template_id: Optional[str] = None
    required_fact_keys: Sequence[str] = field(default_factory=tuple)
    fact_source_refs: Sequence[str] = field(default_factory=tuple)
    fact_freshness_policy: Optional[str] = None
    fallback_when_fact_missing: Optional[str] = None
    bot_permission: str = BOT_PERMISSION_DRAFT_ONLY
    manager_handoff_reason: Optional[str] = None
    rop_review_priority: str = "medium"
    metadata: Mapping[str, Any] = field(default_factory=dict)
    question_class_id: Optional[str] = None

    def __post_init__(self) -> None:
        tenant_id = normalize_key(self.tenant_id, "tenant_id")
        class_key = require_text(self.class_key, "class_key")
        class_id = optional_text(self.question_class_id) or stable_question_class_id(
            tenant_id=tenant_id,
            class_key=class_key,
        )
        if self.count_total < 0 or self.count_calls < 0 or self.count_telegram < 0 or self.count_email < 0:
            raise ValueError("question class counts must not be negative")
        if self.first_seen_at is not None:
            require_timezone(self.first_seen_at, "first_seen_at")
        if self.last_seen_at is not None:
            require_timezone(self.last_seen_at, "last_seen_at")
        if self.first_seen_at and self.last_seen_at and self.last_seen_at < self.first_seen_at:
            raise ValueError("last_seen_at must be greater than or equal to first_seen_at")
        object.__setattr__(self, "tenant_id", tenant_id)
        object.__setattr__(self, "canonical_question", require_text(self.canonical_question, "canonical_question"))
        object.__setattr__(self, "narrow_scope", require_text(self.narrow_scope, "narrow_scope"))
        object.__setattr__(self, "class_key", class_key)
        object.__setattr__(self, "exclusions", str(self.exclusions or "").strip())
        object.__setattr__(self, "examples_redacted", normalize_sequence(self.examples_redacted, "examples_redacted"))
        object.__setattr__(self, "products", normalize_sequence(self.products, "products"))
        object.__setattr__(self, "grades", normalize_sequence(self.grades, "grades"))
        object.__setattr__(self, "subjects", normalize_sequence(self.subjects, "subjects"))
        object.__setattr__(self, "answer_status", normalize_answer_status(self.answer_status))
        object.__setattr__(self, "answer_template_id", optional_text(self.answer_template_id))
        object.__setattr__(self, "required_fact_keys", normalize_sequence(self.required_fact_keys, "required_fact_keys"))
        object.__setattr__(self, "fact_source_refs", normalize_sequence(self.fact_source_refs, "fact_source_refs"))
        object.__setattr__(self, "fact_freshness_policy", optional_text(self.fact_freshness_policy))
        object.__setattr__(self, "fallback_when_fact_missing", optional_text(self.fallback_when_fact_missing))
        object.__setattr__(self, "bot_permission", normalize_bot_permission(self.bot_permission))
        object.__setattr__(self, "manager_handoff_reason", optional_text(self.manager_handoff_reason))
        object.__setattr__(self, "rop_review_priority", normalize_key(self.rop_review_priority, "rop_review_priority"))
        object.__setattr__(self, "metadata", dict(self.metadata))
        object.__setattr__(self, "question_class_id", class_id)

    def to_json_dict(self) -> Mapping[str, Any]:
        return {
            "schema_version": QUESTION_CATALOG_CONTRACTS_SCHEMA_VERSION,
            "question_class_id": self.question_class_id,
            "tenant_id": self.tenant_id,
            "canonical_question": self.canonical_question,
            "narrow_scope": self.narrow_scope,
            "class_key": self.class_key,
            "exclusions": self.exclusions,
            "examples_redacted": list(self.examples_redacted),
            "count_total": self.count_total,
            "count_calls": self.count_calls,
            "count_telegram": self.count_telegram,
            "count_email": self.count_email,
            "first_seen_at": _dt_to_json(self.first_seen_at),
            "last_seen_at": _dt_to_json(self.last_seen_at),
            "products": list(self.products),
            "grades": list(self.grades),
            "subjects": list(self.subjects),
            "answer_status": self.answer_status,
            "answer_template_id": self.answer_template_id,
            "required_fact_keys": list(self.required_fact_keys),
            "fact_source_refs": list(self.fact_source_refs),
            "fact_freshness_policy": self.fact_freshness_policy,
            "fallback_when_fact_missing": self.fallback_when_fact_missing,
            "bot_permission": self.bot_permission,
            "manager_handoff_reason": self.manager_handoff_reason,
            "rop_review_priority": self.rop_review_priority,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class AnswerTemplate:
    tenant_id: str
    question_class_id: str
    template_text: str
    required_fact_keys: Sequence[str] = field(default_factory=tuple)
    approval_status: str = ANSWER_STATUS_DRAFT_NEEDS_REVIEW
    bot_permission_if_facts_fresh: str = BOT_PERMISSION_DRAFT_ONLY
    fallback_when_fact_missing: str = "Передать менеджеру и не называть конкретные условия."
    answer_template_id: Optional[str] = None

    def __post_init__(self) -> None:
        tenant_id = normalize_key(self.tenant_id, "tenant_id")
        text = require_text(self.template_text, "template_text")
        template_id = optional_text(self.answer_template_id) or stable_answer_template_id(
            tenant_id=tenant_id,
            question_class_id=self.question_class_id,
            template_text=text,
        )
        object.__setattr__(self, "tenant_id", tenant_id)
        object.__setattr__(self, "question_class_id", require_text(self.question_class_id, "question_class_id"))
        object.__setattr__(self, "template_text", text)
        object.__setattr__(self, "required_fact_keys", normalize_sequence(self.required_fact_keys, "required_fact_keys"))
        object.__setattr__(self, "approval_status", normalize_answer_status(self.approval_status))
        object.__setattr__(
            self,
            "bot_permission_if_facts_fresh",
            normalize_bot_permission(self.bot_permission_if_facts_fresh),
        )
        object.__setattr__(self, "fallback_when_fact_missing", require_text(self.fallback_when_fact_missing, "fallback_when_fact_missing"))
        object.__setattr__(self, "answer_template_id", template_id)

    def to_json_dict(self) -> Mapping[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CurrentFactSource:
    source_id: str
    fact_types: Sequence[str]
    path: str
    owner: str = "manual_update_required"
    last_updated_at: Optional[str] = None
    freshness_policy: str = "manual_check_required_before_bot_answer"
    usable_for_bot: bool = False
    notes: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_id", normalize_key(self.source_id, "source_id"))
        object.__setattr__(self, "fact_types", normalize_sequence(self.fact_types, "fact_types"))
        object.__setattr__(self, "path", require_text(self.path, "path"))
        object.__setattr__(self, "owner", require_text(self.owner, "owner"))
        object.__setattr__(self, "last_updated_at", optional_text(self.last_updated_at))
        object.__setattr__(self, "freshness_policy", require_text(self.freshness_policy, "freshness_policy"))
        object.__setattr__(self, "notes", str(self.notes or "").strip())

    def to_json_dict(self) -> Mapping[str, Any]:
        return asdict(self)


def question_catalog_safety_contract() -> Mapping[str, Any]:
    return {
        "schema_version": QUESTION_CATALOG_SAFETY_SCHEMA_VERSION,
        "read_only_source_systems": True,
        "write_crm": False,
        "write_tallanto": False,
        "send_email": False,
        "send_messenger": False,
        "live_send": False,
        "run_asr": False,
        "run_ra": False,
        "write_runtime_db": False,
        "mutate_stable_runtime": False,
        "delete_source_artifacts": False,
        "store_raw_personal_data_in_outputs": False,
        "dynamic_facts_required_for_time_sensitive_answers": True,
    }


def assert_question_catalog_safety_contract(contract: Mapping[str, Any]) -> None:
    for key in (
        "write_crm",
        "write_tallanto",
        "send_email",
        "send_messenger",
        "live_send",
        "run_asr",
        "run_ra",
        "write_runtime_db",
        "mutate_stable_runtime",
        "delete_source_artifacts",
        "store_raw_personal_data_in_outputs",
    ):
        if contract.get(key) is not False:
            raise ValueError(f"question catalog safety requires {key}=False")
    if contract.get("read_only_source_systems") is not True:
        raise ValueError("question catalog safety requires read_only_source_systems=True")
    if contract.get("dynamic_facts_required_for_time_sensitive_answers") is not True:
        raise ValueError("question catalog safety requires dynamic facts for time-sensitive answers")


def question_catalog_contract_inventory() -> Mapping[str, Any]:
    return {
        "schema_version": QUESTION_CATALOG_CONTRACTS_SCHEMA_VERSION,
        "sources": sorted(_ALLOWED_SOURCES),
        "answer_statuses": sorted(_ALLOWED_STATUSES),
        "bot_permissions": sorted(_ALLOWED_BOT_PERMISSIONS),
        "fact_types": sorted(
            {
                FACT_TYPE_PRICE,
                FACT_TYPE_SCHEDULE,
                FACT_TYPE_LOCATION,
                FACT_TYPE_DISCOUNT,
                FACT_TYPE_INSTALLMENT,
                FACT_TYPE_TRIAL,
                FACT_TYPE_DOCUMENTS,
                FACT_TYPE_PROGRAM,
            }
        ),
        "safety": question_catalog_safety_contract(),
    }
