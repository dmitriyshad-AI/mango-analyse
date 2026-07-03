from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence


SEMANTIC_READING_CLASSES_ENV = "TELEGRAM_SEMANTIC_READING_CLASSES"
SEMANTIC_READING_SCHEMA_VERSION = "semantic_reading_v1_2026_07_03"
SEMANTIC_READING_SLOT_SOURCE = "semantic_reading_llm"

ALLOWED_SEMANTIC_READING_CLASSES = frozenset({"sense_seats", "off_topic", "slots_gsf"})
ALLOWED_SEMANTIC_READING_SOURCES = frozenset({"inline", "posthoc"})

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
        for candidate in (key, key.lower(), "semantic_reading_classes"):
            if candidate in context:
                return context.get(candidate)
    return os.getenv(key, default)


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


def _normalize_history_texts(texts: Sequence[str]) -> str:
    client_texts = []
    for item in texts:
        text = str(item or "").strip()
        lowered = text.casefold()
        if lowered.startswith(("ответ:", "бот:", "bot:", "assistant:")):
            continue
        if lowered.startswith(("клиент:", "user:", "client:")) and ":" in text:
            text = text.split(":", 1)[1].strip()
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
        if str(raw_text or "").strip().casefold().startswith(("ответ:", "бот:", "bot:", "assistant:")):
            continue
        text = _slot_floor_text(raw_text)
        if not text:
            continue
        if any(marker in text for marker in ("закончил", "закончила", "окончил", "окончила", "перешел", "перешла")):
            continue
        small_numbers = [int(item) for item in _digit_groups(text) if item.isdigit() and 1 <= int(item) <= 11]
        if len(set(small_numbers)) > 1 and "класс" in text:
            continue
        compact = text.replace("-", " ")
        needles = (
            f"{grade} класс",
            f"{grade} классе",
            f"{grade} класса",
            f"{grade}го класса",
            f"{grade}й класс",
            f"{grade} го класса",
            f"{grade} й класс",
        )
        if any(needle in compact for needle in needles):
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
        original = str(raw_text or "").strip()
        lowered = original.casefold()
        if lowered.startswith(("ответ:", "бот:", "bot:", "assistant:")):
            continue
        if lowered.startswith(("клиент:", "user:", "client:")) and ":" in original:
            original = original.split(":", 1)[1].strip()
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
