from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence


SEMANTIC_READING_CLASSES_ENV = "TELEGRAM_SEMANTIC_READING_CLASSES"
READING_APPLY_CLASSES_ENV = "TELEGRAM_READING_APPLY_CLASSES"
PILOT_PROFILE_DEFAULT_READING_CLASSES = "sense_seats,slots_gsf,off_topic,intent_actions,live_status_read"
PILOT_PROFILE_DEFAULT_APPLY_CLASSES = "live_status_read/conversation_intent_plan"
SEMANTIC_READING_SCHEMA_VERSION = "semantic_reading_v1_2026_07_03"
SEMANTIC_READING_TRACE_SCHEMA_VERSION = "semantic_reading_trace_v1_2026_07_03"
SEMANTIC_READING_SLOT_SOURCE = "semantic_reading_llm"

ALLOWED_SEMANTIC_READING_CLASSES = frozenset(
    {
        "sense_seats",
        "off_topic",
        "slots_gsf",
        "intent_actions",
        "route_templates",
        "rewrite_quality",
        "post_semantics",
        "live_status_read",
        "reask_read",
        "roles_read",
    }
)
ALLOWED_READING_APPLY_CLASSES = frozenset(
    {
        "route_templates/autonomy_matrix",
        "live_status_read/conversation_intent_plan",
    }
)
ALLOWED_SEMANTIC_READING_SOURCES = frozenset({"inline", "posthoc"})
SEMANTIC_READING_DECISION_CONFIDENCE = 0.70

_SUBJECT_ALIASES = {
    "физика": ("физик", "physics"),
    "математика": ("математ", "math"),
    "информатика": ("информат", "программ", "coding", "computer"),
    "русский язык": ("русск",),
    "английский язык": ("англ", "english"),
    "химия": ("хими",),
    "биология": ("биолог",),
    "ИИ": ("ии", "искусственный интеллект", "ai"),
}
_FORMAT_ALIASES = {
    "онлайн": ("онлайн", "online", "дистанц", "удален", "удалён", "из дома"),
    "очно": ("очно", "очный", "очная", "офлайн", "offline", "в центр", "в центре", "приезжать"),
}


def _context_value(context: Optional[Mapping[str, Any]], key: str, default: Any = "") -> Any:
    if isinstance(context, Mapping):
        candidates = [key, key.lower()]
        if key == SEMANTIC_READING_CLASSES_ENV:
            candidates.append("semantic_reading_classes")
        for candidate in candidates:
            if candidate in context:
                return context.get(candidate)
    if key in os.environ:
        return os.getenv(key, default)
    if key == SEMANTIC_READING_CLASSES_ENV and _pilot_profile_default_reading_classes_enabled(context):
        return PILOT_PROFILE_DEFAULT_READING_CLASSES
    if key == READING_APPLY_CLASSES_ENV and _pilot_profile_default_reading_classes_enabled(context):
        return PILOT_PROFILE_DEFAULT_APPLY_CLASSES
    return default


def _pilot_profile_default_reading_classes_enabled(context: Optional[Mapping[str, Any]]) -> bool:
    try:
        from mango_mvp.channels.subscription_llm_parts.support import _pilot_gold_profile_enabled
    except Exception:
        return False
    return _pilot_gold_profile_enabled(context)


def _csv_values(value: Any) -> frozenset[str]:
    if isinstance(value, str):
        raw_items = value.split(",")
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        raw_items = [str(item) for item in value]
    else:
        raw_items = []
    return frozenset(item.strip().casefold() for item in raw_items if item.strip())


def enabled_classes(context: Optional[Mapping[str, Any]] = None) -> frozenset[str]:
    return _csv_values(_context_value(context, SEMANTIC_READING_CLASSES_ENV, "")) & ALLOWED_SEMANTIC_READING_CLASSES


def reading_class_enabled(context: Optional[Mapping[str, Any]], name: str) -> bool:
    return str(name or "").strip().casefold() in enabled_classes(context)


def apply_classes(context: Optional[Mapping[str, Any]] = None) -> frozenset[str]:
    return _csv_values(_context_value(context, READING_APPLY_CLASSES_ENV, "")) & ALLOWED_READING_APPLY_CLASSES


def reading_apply_class_enabled(context: Optional[Mapping[str, Any]], name: str) -> bool:
    normalized = str(name or "").strip().casefold()
    base_class = normalized.split("/", 1)[0]
    return base_class in enabled_classes(context) and normalized in apply_classes(context)


def _clamp01(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if number < 0.0:
        return 0.0
    if number > 1.0:
        return 1.0
    return number


def _clean_text(value: Any, *, limit: int = 160) -> str:
    return " ".join(str(value or "").split())[:limit]


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _clean_trace_text(value: Any, *, limit: int = 160) -> str:
    return " ".join(str(value or "").split())[:limit]


def _clean_trace_fields(value: Any) -> list[str]:
    if isinstance(value, str):
        raw_items = value.split(",")
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        raw_items = [str(item) for item in value]
    else:
        raw_items = []
    out: list[str] = []
    for raw in raw_items:
        item = _clean_trace_text(raw, limit=80)
        if item and item not in out:
            out.append(item)
    return out[:12]


def _trace_text_hash(value: Any) -> str:
    text = str(value or "")
    if not text:
        return ""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def semantic_reading_transition_metadata(
    *,
    stage: str,
    draft_before: Any = "",
    draft_after: Any = "",
    text_replacement: bool = False,
    legacy_decision: str = "",
    frame_decision: str = "",
    chosen: str = "",
    extra: Optional[Mapping[str, Any]] = None,
) -> Mapping[str, Any]:
    payload: dict[str, Any] = {
        "stage": _clean_trace_text(stage, limit=80),
        "draft_before_hash": _trace_text_hash(draft_before),
        "draft_after_hash": _trace_text_hash(draft_after),
        "text_replacement": bool(text_replacement),
        "legacy_decision": _clean_trace_text(legacy_decision, limit=120),
        "frame_decision": _clean_trace_text(frame_decision, limit=120),
        "chosen": _clean_trace_text(chosen, limit=120),
    }
    if isinstance(extra, Mapping):
        for key, value in extra.items():
            payload[_clean_trace_text(key, limit=80)] = value
    return payload


def semantic_reading_trace_record(
    *,
    reading_class: str,
    enabled: bool,
    status: str,
    decision: str = "",
    reason: str = "",
    source: str = "",
    confidence: Any = 0.0,
    changed_fields: Sequence[str] = (),
    conflicts: Sequence[str] = (),
    metadata: Optional[Mapping[str, Any]] = None,
) -> Mapping[str, Any]:
    payload = dict(metadata or {}) if isinstance(metadata, Mapping) else {}
    return {
        "schema_version": SEMANTIC_READING_TRACE_SCHEMA_VERSION,
        "class": _clean_trace_text(reading_class, limit=80),
        "enabled": bool(enabled),
        "status": _clean_trace_text(status, limit=80),
        "decision": _clean_trace_text(decision, limit=120),
        "reason": _clean_trace_text(reason, limit=160),
        "source": _clean_trace_text(source, limit=40),
        "confidence": _clamp01(confidence),
        "changed_fields": _clean_trace_fields(changed_fields),
        "conflict_with": _clean_trace_fields(conflicts),
        "metadata": payload,
    }


def append_reading_trace_record(
    metadata: Mapping[str, Any],
    record: Mapping[str, Any],
    *,
    max_records: int = 12,
) -> dict[str, Any]:
    result = dict(metadata or {})
    raw_records = result.get("semantic_reading_trace")
    records = [dict(item) for item in raw_records if isinstance(item, Mapping)] if isinstance(raw_records, Sequence) and not isinstance(raw_records, (str, bytes, bytearray)) else []
    records.append(dict(record))
    if len(records) > max_records:
        records = [
            *records[: max_records - 1],
            semantic_reading_trace_record(
                reading_class="trace",
                enabled=True,
                status="truncated",
                reason=f"{len(records) - (max_records - 1)} records omitted",
            ),
        ]
    result["semantic_reading_trace"] = records
    direct = dict(result.get("direct_path") or {}) if isinstance(result.get("direct_path"), Mapping) else {}
    if direct:
        direct["semantic_reading_trace"] = records
        result["direct_path"] = direct
    return result


def finalize_reading_trace_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(metadata or {})
    direct = dict(result.get("direct_path") or {}) if isinstance(result.get("direct_path"), Mapping) else {}
    raw_records = result.get("semantic_reading_trace") or direct.get("semantic_reading_trace")
    if not isinstance(raw_records, Sequence) or isinstance(raw_records, (str, bytes, bytearray)):
        return result
    records = [dict(item) for item in raw_records if isinstance(item, Mapping)]
    if not records:
        result.pop("semantic_reading_trace", None)
        if direct:
            direct.pop("semantic_reading_trace", None)
            result["direct_path"] = direct
        return result
    result["semantic_reading_trace"] = records
    if direct:
        direct["semantic_reading_trace"] = records
        result["direct_path"] = direct
    return result


def semantic_frame_from_metadata(metadata: Mapping[str, Any]) -> Mapping[str, Any]:
    direct = _mapping(metadata.get("direct_path"))
    for container in (metadata, direct):
        frame = container.get("semantic_frame")
        if isinstance(frame, Mapping):
            return frame
        frame = container.get("semantic_frame_shadow")
        if isinstance(frame, Mapping):
            return frame
    return {}


def _source_from_metadata(frame: Mapping[str, Any], metadata: Mapping[str, Any]) -> str:
    source = str(frame.get("source") or "").strip().casefold()
    if source in ALLOWED_SEMANTIC_READING_SOURCES:
        return source
    direct = _mapping(metadata.get("direct_path"))
    posthoc = metadata.get("semantic_frame_posthoc_shadow")
    if not isinstance(posthoc, Mapping):
        posthoc = direct.get("semantic_frame_posthoc_shadow")
    if isinstance(posthoc, Mapping) and str(posthoc.get("status") or "").strip().casefold() == "ok":
        return "posthoc"
    return "inline"


@dataclass(frozen=True)
class SemanticReading:
    source: str
    primary_intent: str = ""
    sense: str = ""
    scope: str = ""
    intent_confidence: float = 0.0
    requested_action: str = ""
    product_grade: str = ""
    product_subject: str = ""
    product_format: str = ""
    product_raw_text: str = ""
    frame_confidence: float = 0.0
    schema_version: str = SEMANTIC_READING_SCHEMA_VERSION

    @staticmethod
    def from_result(result: Any, *, context: Optional[Mapping[str, Any]] = None) -> Optional["SemanticReading"]:
        del context
        metadata = _mapping(getattr(result, "metadata", {}))
        model_intent = _mapping(metadata.get("direct_path_model_intent"))
        if not model_intent:
            model_intent = _mapping(_mapping(metadata.get("direct_path")).get("model_intent"))
        frame = semantic_frame_from_metadata(metadata)
        if not model_intent and not frame:
            return None
        requested_product = _mapping(frame.get("requested_product"))
        return SemanticReading(
            source=_source_from_metadata(frame, metadata),
            primary_intent=_clean_text(model_intent.get("primary_intent"), limit=80),
            sense=_clean_text(model_intent.get("sense"), limit=80),
            scope=_clean_text(model_intent.get("scope"), limit=120),
            intent_confidence=_clamp01(model_intent.get("confidence")),
            requested_action=_clean_text(frame.get("requested_action"), limit=120),
            product_grade=_clean_text(requested_product.get("grade"), limit=40),
            product_subject=_clean_text(requested_product.get("subject"), limit=80),
            product_format=_clean_text(requested_product.get("format"), limit=80),
            product_raw_text=_clean_text(requested_product.get("raw_text"), limit=160),
            frame_confidence=_clamp01(frame.get("confidence")),
        )

    def to_memory_dict(self) -> Mapping[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source": self.source,
            "primary_intent": self.primary_intent,
            "sense": self.sense,
            "scope": self.scope,
            "intent_confidence": self.intent_confidence,
            "requested_action": self.requested_action,
            "product_grade": self.product_grade,
            "product_subject": self.product_subject,
            "product_format": self.product_format,
            "product_raw_text": self.product_raw_text,
            "frame_confidence": self.frame_confidence,
        }


def _client_authored_texts(value: str) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return []
    lowered = text.casefold()
    if lowered.startswith(("ответ:", "бот:", "bot:", "assistant:")):
        return []
    for prefix in ("клиент:", "user:", "client:", "turn_msg:"):
        if lowered.startswith(prefix) and ":" in text:
            return [text.split(":", 1)[1].strip()]
    if lowered.startswith("history/persona:") and ":" in text:
        body = text.split(":", 1)[1].strip()
        if not any(marker in body for marker in ("Клиент:", "клиент:", "Client:", "client:")):
            return []
        pieces: list[str] = []
        for marker in ("Клиент:", "клиент:", "Client:", "client:"):
            if marker not in body:
                continue
            for part in body.split(marker)[1:]:
                end = len(part)
                for stop in ("Ответ:", "ответ:", "Бот:", "бот:", "Assistant:", "assistant:", "Client:", "client:"):
                    index = part.find(stop)
                    if index >= 0:
                        end = min(end, index)
                cleaned = part[:end].strip(" ;…")
                if cleaned:
                    pieces.append(cleaned)
        return pieces
    return [text]


def _normalize_history_texts(texts: Sequence[str]) -> str:
    client_texts = []
    for item in texts:
        for text in _client_authored_texts(str(item or "")):
            client_texts.append(" ".join(text.casefold().replace("ё", "е").split()))
    return " ".join(client_texts)


def _digit_groups(value: str) -> list[str]:
    groups: list[str] = []
    current: list[str] = []
    for char in str(value or ""):
        if char.isdigit():
            current.append(char)
        elif current:
            groups.append("".join(current))
            current = []
    if current:
        groups.append("".join(current))
    return groups


def _slot_floor_text(value: str) -> str:
    text = str(value or "").casefold().replace("ё", "е")
    for char in ",.;:!?()[]{}«»\"'":
        text = text.replace(char, " ")
    return " ".join(text.split())


def _class_grade_hits(text: str) -> set[str]:
    compact = _slot_floor_text(text).replace("-", " ")
    hits: set[str] = set()
    for grade in range(1, 12):
        value = str(grade)
        needles = (
            f"{value} класс",
            f"{value} классе",
            f"{value} класса",
            f"{value}го класса",
            f"{value}й класс",
            f"{value} го класса",
            f"{value} й класс",
        )
        if any(needle in compact for needle in needles):
            hits.add(value)
    return hits


def _normalize_grade(value: str) -> str:
    text = _clean_text(value, limit=40).casefold().replace("ё", "е")
    numbers = [item for item in _digit_groups(text) if item.isdigit()]
    if len(numbers) != 1:
        return ""
    try:
        grade = int(numbers[0])
    except ValueError:
        return ""
    if not 1 <= grade <= 11:
        return ""
    if text == str(grade) or "класс" in text or "grade" in text:
        return str(grade)
    return ""


def _normalize_alias(value: str, aliases: Mapping[str, Sequence[str]]) -> str:
    text = _clean_text(value, limit=80).casefold().replace("ё", "е")
    if not text:
        return ""
    if aliases is _SUBJECT_ALIASES and text in {"ит", "it"}:
        return "информатика"
    for canonical, needles in aliases.items():
        canonical_text = str(canonical).casefold().replace("ё", "е")
        if canonical_text in text or any(str(needle).casefold().replace("ё", "е") in text for needle in needles):
            return canonical
    return ""


def _value_supported_by_history(value: str, history_texts: Sequence[str]) -> bool:
    if not value:
        return False
    history = _normalize_history_texts(history_texts)
    if not history:
        return False
    return _clean_text(value, limit=80).casefold().replace("ё", "е") in history


def _history_supports_grade(grade: str, history_texts: Sequence[str]) -> bool:
    if not grade:
        return False
    for raw_text in history_texts:
        for client_text in _client_authored_texts(str(raw_text or "")):
            text = _slot_floor_text(client_text)
            if not text:
                continue
            if any(marker in text for marker in ("закончил", "закончила", "окончил", "окончила")):
                continue
            class_grade_hits = _class_grade_hits(text)
            if len(class_grade_hits) > 1:
                continue
            if grade in class_grade_hits:
                return True
    return False


def _history_supports_alias(value: str, aliases: Mapping[str, Sequence[str]], history_texts: Sequence[str]) -> bool:
    if not value:
        return False
    needles = (value, *aliases.get(value, ()))
    history = _normalize_history_texts(history_texts)
    if aliases is _SUBJECT_ALIASES and value == "информатика":
        if " ит " in f" {history} " or " it " in f" {history} ":
            return True
    return any(str(needle or "").casefold().replace("ё", "е") in history for needle in needles)


def _alias_hits(text: str, aliases: Mapping[str, Sequence[str]]) -> set[str]:
    normalized = _slot_floor_text(text)
    hits: set[str] = set()
    for canonical, needles in aliases.items():
        canonical_text = str(canonical).casefold().replace("ё", "е")
        if canonical_text in normalized or any(str(needle).casefold().replace("ё", "е") in normalized for needle in needles):
            hits.add(str(canonical))
    if aliases is _SUBJECT_ALIASES and (" ит " in f" {normalized} " or " it " in f" {normalized} "):
        hits.add("информатика")
    return hits


def _history_has_multi_alias_choice(history_texts: Sequence[str], aliases: Mapping[str, Sequence[str]]) -> bool:
    for raw_text in history_texts:
        for original in _client_authored_texts(str(raw_text or "")):
            normalized = _slot_floor_text(original)
            raw_normalized = " ".join(original.casefold().replace("ё", "е").split())
            if not normalized:
                continue
            hits = _alias_hits(normalized, aliases)
            if len(hits) >= 2 and any(separator in raw_normalized for separator in (" или ", " либо ", " и ", " / ", "/", ",")):
                return True
    return False


def slot_candidates_from_reading(
    reading: Optional[SemanticReading],
    *,
    history_texts: Sequence[str] = (),
    confidence_threshold: float = 0.70,
) -> Mapping[str, Mapping[str, Any]]:
    if reading is None or reading.source != "inline" or reading.frame_confidence < confidence_threshold:
        return {}
    raw_values = {
        "grade": reading.product_grade,
        "subject": reading.product_subject,
        "format": reading.product_format,
    }
    normalized = {
        "grade": _normalize_grade(raw_values["grade"]),
        "subject": _normalize_alias(raw_values["subject"], _SUBJECT_ALIASES),
        "format": _normalize_alias(raw_values["format"], _FORMAT_ALIASES),
    }
    out: dict[str, Mapping[str, Any]] = {}
    subject_is_ambiguous = _history_has_multi_alias_choice(history_texts, _SUBJECT_ALIASES)
    format_is_ambiguous = _history_has_multi_alias_choice(history_texts, _FORMAT_ALIASES)
    for key, value in normalized.items():
        if not value:
            continue
        raw_value = raw_values[key]
        if key == "grade":
            supported = _history_supports_grade(value, history_texts)
        elif key == "subject":
            if subject_is_ambiguous:
                continue
            supported = _history_supports_alias(value, _SUBJECT_ALIASES, history_texts) or _value_supported_by_history(
                raw_value, history_texts
            )
        elif key == "format":
            if format_is_ambiguous:
                continue
            supported = _history_supports_alias(value, _FORMAT_ALIASES, history_texts) or _value_supported_by_history(
                raw_value, history_texts
            )
        else:
            supported = _value_supported_by_history(value, history_texts) or _value_supported_by_history(
                raw_value, history_texts
            )
        if not supported:
            continue
        out[key] = {
            "value": value,
            "source_name": SEMANTIC_READING_SLOT_SOURCE,
            "confidence": reading.frame_confidence,
            "evidence": reading.product_raw_text or raw_value,
        }
    return out


def sense_seats_reading_decision(reading: Optional[SemanticReading], client_text: str = "") -> str:
    if reading is None or reading.source != "inline" or reading.intent_confidence < SEMANTIC_READING_DECISION_CONFIDENCE:
        return ""
    primary = str(reading.primary_intent or "").strip().casefold()
    sense = str(reading.sense or "").strip().casefold()
    action = str(reading.requested_action or "").strip().casefold()
    scope = str(reading.scope or "").strip().casefold()
    if primary in {"live_availability", "availability"} or sense in {
        "seats",
        "seat_availability",
        "availability",
        "live_status",
        "booking",
    }:
        return "seats"
    if action in {"check_availability", "enroll"} and "мест" in scope:
        return "seats"
    if "мест" in str(client_text or "").casefold().replace("ё", "е"):
        return "not_seats"
    return ""


def off_topic_reading_decision(reading: Optional[SemanticReading]) -> str:
    if reading is None or reading.source != "inline" or reading.intent_confidence < SEMANTIC_READING_DECISION_CONFIDENCE:
        return ""
    primary = str(reading.primary_intent or "").strip().casefold()
    return "off_topic" if primary == "off_topic" else "not_off_topic"


def slots_reading_candidates(
    reading: Optional[SemanticReading],
    history_texts: Sequence[str] = (),
    *,
    confidence_threshold: float = 0.70,
) -> Mapping[str, Mapping[str, Any]]:
    return slot_candidates_from_reading(
        reading,
        history_texts=history_texts,
        confidence_threshold=confidence_threshold,
    )
