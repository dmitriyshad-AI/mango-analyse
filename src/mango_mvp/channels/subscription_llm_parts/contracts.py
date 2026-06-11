from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence


SUBSCRIPTION_LLM_SCHEMA_VERSION = "subscription_llm_draft_v1_2026_05_16"


SAFE_FALLBACK_DRAFT_TEXT = "Чтобы не ошибиться, передам вопрос менеджеру — он сверит детали и вернётся с ответом."


INTERNAL_SERVICE_MARKER_RE = re.compile(
    r"\[[^\]\n]{0,220}?(?:\bsource(?:_id)?\s*[:=]|\bfreshness\s*[:=]|source:[A-Za-z0-9_:\-]+|fact:[A-Za-z0-9_:\-]+|kc_chunk:[A-Za-z0-9_:\-]+|kb_release_[A-Za-z0-9_\-]+|product_data/[^\]\s]+|/Users/[^\]\s]+)[^\]\n]{0,260}\]\s*",
    re.I,
)


INTERNAL_SERVICE_TOKEN_RE = re.compile(
    r"\b(?:source|source_id|fact_id|trace_id|freshness)\s*[:=]\s*[^\s;\],.]+|source:[A-Za-z0-9_:\-]+|fact:[A-Za-z0-9_:\-]+|kc_chunk:[A-Za-z0-9_:\-]+|kb_release_[A-Za-z0-9_\-]+|product_data/[^\s;\],.]+|/Users/[^\s;\],.]+",
    re.I,
)


INTERNAL_SCAFFOLD_PREFIX_RE = re.compile(
    r"^\s*(?:[^:\n]{1,80}:\s*)?(?:черновик\s+)?для\s+ситуации\s+[«\"][^»\"\n]{1,160}[»\"]\s*:\s*",
    re.I,
)


INTERNAL_PROMPT_DIRECTIVE_PREFIX_RE = re.compile(
    r"^\s*без\s+(?:обещан\w+|давлен\w+)[^:\n]{0,180}:\s*",
    re.I,
)


INTERNAL_PROMPT_DIRECTIVE_ANYWHERE_RE = re.compile(
    r"\s*(?:по\s+вашей\s+ситуации\s+лучше\s+опираться\s+на\s+подтвержд[её]нные\s+условия,\s*)?"
    r"без\s+обещан\w+[^:\n]{0,120}:\s*",
    re.I,
)


INTERNAL_CLIENT_SAFE_JARGON_RE = re.compile(
    r"(?:нет\s+)?client[-\s]?safe\s+факт[^\n.?!]*(?:[.?!]|$)|\bclient[-\s]?safe\b",
    re.I,
)


INTERNAL_RUNTIME_LIMIT_JARGON_RE = re.compile(
    r"(?:^|(?<=[.?!\n]))\s*(?:"
    r"лимит(?:ы)?\s+(?:codex|кодекса|сессии|контекста)[^.?!\n]{0,180}|"
    r"(?:осталось|остаток)\s+\d{1,5}\s+(?:сообщен\w+|запрос\w+|токен\w+|лимит\w+|контекст\w+)[^.?!\n]{0,180}|"
    r"(?:сессия|контекст|лимит)\s+(?:заканчива\w+|исчерпан\w+|подход\w+\s+к\s+концу)[^.?!\n]{0,180}|"
    r"(?:заканчива\w+|исчерпан\w+|подход\w+\s+к\s+концу)\s+(?:сессия|контекст|лимит)[^.?!\n]{0,180}"
    r")(?:[.?!]|$)",
    re.I,
)


INTERNAL_REGEN_EDIT_COMMENT_RE = re.compile(
    r"(?:^|(?<=\n)|(?<=[.?!]))\s*(?:заменяю|переписываю|исправляю|меняю)\s+(?:только\s+)?(?:этот\s+|данный\s+)?"
    r"(?:абзац|фрагмент|текст|ответ)[^:\n]{0,160}:\s*"
    r"|(?:^|(?<=\n)|(?<=[.?!]))\s*(?:остальн\w+\s+текст|остальные\s+абзацы|остальное)\s+"
    r"(?:оставляю\s+|оставь\s+)?без\s+изменен\w+\s*[.?!:;]?\s*",
    re.I,
)


INTERNAL_CLIENT_INSTRUCTION_RE = re.compile(
    r"(?:\bповторять\s+(?:их\s+)?не\s+нужно\b|\bне\s+упоминай\w*\b|"
    r"\bесли\b[^.?!\n]{0,140}\bуже\s+есть\s+в\s+диалоге\b[^.?!\n]{0,140})",
    re.I,
)


INTERNAL_MANAGER_DRAFT_RE = re.compile(
    r"(?:автономн\w+\s+ответ\s+не\s+требуется|дополнительн\w+\s+ответ\s+клиенту\s+сейчас\s+не\s+нужен|если\s+менеджер\s+решит\s+ответить|безопасн\w+\s+вариант|без\s+служебн\w+\s+помет\w+|клиент\s+(?:понял|подтвердил|взял\s+пауз))",
    re.I,
)


INTERNAL_SAFE_VARIANT_RE = re.compile(
    r"безопасн\w+\s+вариант\s*:\s*[«\"](?P<text>.+?)[»\"]\s*$",
    re.I | re.S,
)


ALLOWED_ROUTES = {"draft_for_manager", "manager_only", "blocked", "bot_answer_self", "bot_answer_self_for_pilot"}


ALLOWED_MESSAGE_TYPES = {"question", "non_question", "context_update", "wait_for_more", "manager_only"}


BASE_SAFETY_FLAGS = ("manager_approval_required", "no_auto_send")


@dataclass(frozen=True)
class SubscriptionDraftResult:
    message_type: str = "question"
    broad_group: str = ""
    topic_id: str = "service:S2_unclear"
    topic_confidence: float = 0.0
    confidence_group: float = 0.0
    alternative_themes: tuple[str, ...] = field(default_factory=tuple)
    risk_level: str = "unknown"
    route: str = "manager_only"
    veto_category: str = ""
    draft_text: str = SAFE_FALLBACK_DRAFT_TEXT
    manager_checklist: tuple[str, ...] = field(default_factory=tuple)
    missing_facts: tuple[str, ...] = field(default_factory=tuple)
    forbidden_promises_detected: tuple[str, ...] = field(default_factory=tuple)
    crm_recommendations: tuple[Mapping[str, Any], ...] = field(default_factory=tuple)
    safety_flags: tuple[str, ...] = BASE_SAFETY_FLAGS
    context_used: tuple[str, ...] = field(default_factory=tuple)
    context_warnings: tuple[str, ...] = field(default_factory=tuple)
    manager_followup_required: bool = False
    manager_followup_deadline: Optional[str] = None
    provider: str = "codex_exec"
    schema_version: str = SUBSCRIPTION_LLM_SCHEMA_VERSION
    raw_response: Optional[str] = None
    error: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        route = str(self.route or "manager_only").strip()
        if route not in ALLOWED_ROUTES:
            route = "manager_only"
        raw_text = str(self.draft_text or "").strip()
        text = strip_internal_service_markers(raw_text) or SAFE_FALLBACK_DRAFT_TEXT
        message_type = str(self.message_type or "question").strip()
        if message_type not in ALLOWED_MESSAGE_TYPES:
            message_type = "manager_only"
        extra_flags = ["internal_metadata_removed_from_draft"] if text != raw_text and raw_text else []
        metadata = dict(self.metadata)
        if extra_flags:
            metadata.setdefault("guarded_original_text", " ".join(raw_text.split())[:500])
            metadata.setdefault("guarded_original_text_guard", "strip_internal_service_markers")
            guards = [str(item) for item in (metadata.get("guarded_original_text_guards") or []) if str(item).strip()]
            if "strip_internal_service_markers" not in guards:
                guards.append("strip_internal_service_markers")
            metadata["guarded_original_text_guards"] = guards[:8]
        flags = tuple(
            dict.fromkeys(
                [
                    *BASE_SAFETY_FLAGS,
                    *(_clean_list(self.safety_flags, max_items=16, max_chars=80)),
                    *extra_flags,
                ]
            )
        )
        object.__setattr__(self, "message_type", message_type)
        object.__setattr__(self, "broad_group", str(self.broad_group or "").strip()[:80])
        object.__setattr__(self, "route", route)
        object.__setattr__(self, "veto_category", str(self.veto_category or "").strip()[:80])
        object.__setattr__(self, "draft_text", text)
        object.__setattr__(self, "topic_id", str(self.topic_id or "service:S2_unclear").strip() or "service:S2_unclear")
        object.__setattr__(self, "topic_confidence", _clamp_float(self.topic_confidence))
        object.__setattr__(self, "confidence_group", _clamp_float(self.confidence_group))
        object.__setattr__(self, "alternative_themes", tuple(_clean_list(self.alternative_themes, max_items=5, max_chars=120)))
        object.__setattr__(self, "risk_level", str(self.risk_level or "unknown").strip()[:80] or "unknown")
        object.__setattr__(self, "manager_checklist", tuple(_clean_list(self.manager_checklist, max_items=12, max_chars=240)))
        object.__setattr__(self, "missing_facts", tuple(_clean_list(self.missing_facts, max_items=12, max_chars=160)))
        object.__setattr__(
            self,
            "forbidden_promises_detected",
            tuple(_clean_list(self.forbidden_promises_detected, max_items=12, max_chars=160)),
        )
        object.__setattr__(self, "crm_recommendations", tuple(_clean_crm_recommendations(self.crm_recommendations)))
        object.__setattr__(self, "safety_flags", flags)
        object.__setattr__(self, "context_used", tuple(_clean_list(self.context_used, max_items=12, max_chars=100)))
        object.__setattr__(self, "context_warnings", tuple(_clean_list(self.context_warnings, max_items=12, max_chars=120)))
        object.__setattr__(self, "metadata", metadata)

    def to_json_dict(self, *, include_raw_response: bool = False) -> Mapping[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "provider": self.provider,
            "message_type": self.message_type,
            "broad_group": self.broad_group,
            "topic_id": self.topic_id,
            "topic_confidence": self.topic_confidence,
            "confidence_theme": self.topic_confidence,
            "confidence_group": self.confidence_group,
            "alternative_themes": list(self.alternative_themes),
            "risk_level": self.risk_level,
            "route": self.route,
            "veto_category": self.veto_category,
            "draft_text": self.draft_text,
            "manager_checklist": list(self.manager_checklist),
            "missing_facts": list(self.missing_facts),
            "forbidden_promises_detected": list(self.forbidden_promises_detected),
            "crm_recommendations": [dict(item) for item in self.crm_recommendations],
            "manager_followup_required": self.manager_followup_required,
            "manager_followup_deadline": self.manager_followup_deadline,
            "safety_flags": list(self.safety_flags),
            "context_used": list(self.context_used),
            "context_warnings": list(self.context_warnings),
            "error": self.error,
            "metadata": dict(self.metadata),
        }
        if include_raw_response:
            payload["raw_response"] = self.raw_response
        return payload


def _normalize_output_sanitizer_text(text: str) -> str:
    value = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t\f\v]+", " ", line).strip() for line in value.split("\n")]
    value = "\n".join(lines)
    value = re.sub(r"\s+([,.;:!?])", r"\1", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def strip_internal_service_markers(text: str) -> str:
    value = str(text or "")
    if not value:
        return ""
    safe_variant = INTERNAL_SAFE_VARIANT_RE.search(value)
    if safe_variant:
        candidate = _normalize_output_sanitizer_text(str(safe_variant.group("text") or ""))
        if candidate and not INTERNAL_MANAGER_DRAFT_RE.search(candidate):
            return candidate.strip()
    if INTERNAL_MANAGER_DRAFT_RE.search(value):
        return ""
    previous = None
    while previous != value:
        previous = value
        value = INTERNAL_SCAFFOLD_PREFIX_RE.sub("", value)
        value = INTERNAL_PROMPT_DIRECTIVE_PREFIX_RE.sub("", value)
        value = value.lstrip()
    if INTERNAL_CLIENT_INSTRUCTION_RE.search(value):
        return ""
    value = INTERNAL_REGEN_EDIT_COMMENT_RE.sub(" ", value)
    value = INTERNAL_PROMPT_DIRECTIVE_ANYWHERE_RE.sub(" ", value)
    value = INTERNAL_CLIENT_SAFE_JARGON_RE.sub(" ", value)
    value = INTERNAL_RUNTIME_LIMIT_JARGON_RE.sub(" ", value)
    value = INTERNAL_SERVICE_MARKER_RE.sub("", value)
    value = INTERNAL_SERVICE_TOKEN_RE.sub("", value)
    if INTERNAL_CLIENT_INSTRUCTION_RE.search(value):
        return ""
    return _normalize_output_sanitizer_text(value)


def _clean_list(value: Any, *, max_items: int, max_chars: int) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        values: Sequence[Any] = [value]
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        values = value
    else:
        return []
    result: list[str] = []
    for item in values:
        text = str(item or "").strip()
        if not text:
            continue
        result.append(" ".join(text.split())[:max_chars])
        if len(result) >= max_items:
            break
    return result


def _clean_crm_recommendations(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    result: list[Mapping[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        recommendation = {
            "target": str(item.get("target") or "").strip()[:80],
            "action": str(item.get("action") or "").strip()[:80],
            "text": str(item.get("text") or "").strip()[:500],
            "requires_manager_approval": True,
        }
        if recommendation["target"] and recommendation["action"] and recommendation["text"]:
            result.append(recommendation)
        if len(result) >= 8:
            break
    return result


def _clamp_float(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return min(1.0, max(0.0, parsed))
