from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

from mango_mvp.insights.sanitizers import (
    has_any_safety_risk,
    sanitize_answer,
)


EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d[\s().\-]*){10,15}(?!\d)")
URL_RE = re.compile(r"\b(?:https?\s*:\s*//|www\.)\S+", re.I)
LONG_NUMBER_RE = re.compile(r"(?<!\d)(?:\d[\s().\-]*){10,}(?!\d)")
PUBLIC_MASK_PHRASES = (
    "действующие правила изменения или отмены услуги",
    "актуальное окно записи",
    "актуальную стоимость",
    "актуальные варианты",
    "удобный контакт",
    "адрес, который подтвердит менеджер",
    "Точные условия менеджер подтвердит по актуальным правилам.",
)


def is_stable_runtime_path(path: Path | str) -> bool:
    return any(part.casefold() == "stable_runtime" for part in Path(path).parts)


def guard_question_catalog_output_path(path: Path | str, *, project_root: Path | str) -> Path:
    resolved = Path(path).resolve(strict=False)
    root = Path(project_root).resolve(strict=False)
    if is_stable_runtime_path(resolved):
        raise ValueError(f"question catalog output must not be under stable_runtime: {resolved}")
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"question catalog output must stay under project root: {root}") from exc
    return resolved


def redact_public_text(value: Any, *, max_chars: int = 500) -> tuple[str, tuple[str, ...]]:
    source = str(value or "").strip()
    sanitized = sanitize_answer(value, mode="bot")
    text, extra_flags = _redact_extra_public_risks(compress_mask_runs(sanitized.text.strip()))
    if source and not text:
        return "[очищено: исходный текст полностью состоял из небезопасных данных]", tuple(
            dict.fromkeys((*sanitized.flags, *extra_flags, "empty_after_public_redaction"))
        )
    if max_chars > 0 and len(text) > max_chars:
        text = text[: max_chars - 1].rstrip() + "..."
    if has_any_safety_risk(text) or contains_raw_contact(text):
        return "[очищено: потенциальные персональные или изменяемые данные]", tuple(
            dict.fromkeys((*sanitized.flags, *extra_flags, "blocked_unresolved_public_risk"))
        )
    return text, tuple(dict.fromkeys((*sanitized.flags, *extra_flags)))


def redact_review_text(value: Any, *, max_chars: int = 700) -> tuple[str, tuple[str, ...]]:
    """Keep examples readable for internal ROP review while removing direct contacts."""

    text = str(value or "").strip()
    flags: list[str] = []
    text = re.sub(r"^(?:re(?:\[\d+\])?:\s*)+", "", text, flags=re.I).strip()
    text = re.sub(r"\s*(?:отправлено\s+из\s+(?:мобильной\s+)?(?:почты\s+)?mail).*", "", text, flags=re.I).strip()
    text = re.sub(r"\s*-{4,}\s*пересылаемое\s+сообщение\s*-{4,}.*", "", text, flags=re.I).strip()
    if EMAIL_RE.search(text):
        text = EMAIL_RE.sub("[email скрыт]", text)
        flags.append("email_redacted")
    if PHONE_RE.search(text):
        text = PHONE_RE.sub("[телефон скрыт]", text)
        flags.append("phone_redacted")
    if URL_RE.search(text):
        text = URL_RE.sub("[ссылка скрыта]", text)
        flags.append("url_redacted")
    if LONG_NUMBER_RE.search(text):
        text = LONG_NUMBER_RE.sub("[длинный номер скрыт]", text)
        flags.append("long_number_redacted")
    text = re.sub(r"\s+", " ", text).strip()
    if max_chars > 0 and len(text) > max_chars:
        text = text[: max_chars - 1].rstrip() + "..."
    if not text and str(value or "").strip():
        return "[очищено: пример состоял из контактных или служебных данных]", tuple(dict.fromkeys((*flags, "empty_after_review_redaction")))
    return text, tuple(dict.fromkeys(flags))


def compress_mask_runs(value: Any) -> str:
    text = str(value or "")
    for phrase in PUBLIC_MASK_PHRASES:
        escaped = re.escape(phrase)
        text = re.sub(rf"(?:{escaped})(?:\s+{escaped})+", phrase, text, flags=re.I)
    text = text.replace("Точные условия менеджер подтвердит по актуальным правилам.", "").strip()
    return re.sub(r"\s+", " ", text).strip()


def score_example_readability(value: Any) -> float:
    text = str(value or "").strip()
    if not text:
        return 0.0
    words = re.findall(r"[a-zа-яё0-9]+", text, re.I)
    if len(words) < 4:
        return 0.1
    score = 1.0
    lowered = text.casefold()
    mask_hits = sum(lowered.count(phrase.casefold()) for phrase in PUBLIC_MASK_PHRASES)
    if mask_hits:
        score -= min(0.75, mask_hits * 0.18)
    if "?" in text:
        score += 0.15
    if 40 <= len(text) <= 240:
        score += 0.15
    if len(text) > 500:
        score -= 0.25
    if re.search(r"\b(?:клиент\s+только\s+сказал|содержательный\s+запрос\s+не\s+сформулирован|возражение/|интент:|label:)\b", lowered, re.I):
        score -= 0.65
    return max(0.0, min(1.0, score))


def contains_raw_contact(value: Any) -> bool:
    text = str(value or "")
    return bool(EMAIL_RE.search(text) or PHONE_RE.search(text) or URL_RE.search(text) or LONG_NUMBER_RE.search(text))


def _redact_extra_public_risks(value: str) -> tuple[str, tuple[str, ...]]:
    text = value
    flags: list[str] = []
    if URL_RE.search(text):
        text = URL_RE.sub("ссылка", text)
        flags.append("url_redacted")
    if LONG_NUMBER_RE.search(text):
        text = LONG_NUMBER_RE.sub("служебный номер", text)
        flags.append("long_number_redacted")
    text = re.sub(r"\s+", " ", text).strip()
    return text, tuple(flags)


def assert_public_text_safe(value: Any, *, field_name: str = "text") -> None:
    text = str(value or "")
    if contains_raw_contact(text) or has_any_safety_risk(text):
        raise ValueError(f"{field_name} contains unsafe public text")


def assert_public_rows_safe(rows: Iterable[dict[str, Any]], *, text_fields: Iterable[str]) -> None:
    fields = tuple(text_fields)
    for row_index, row in enumerate(rows, start=1):
        for field in fields:
            assert_public_text_safe(row.get(field, ""), field_name=f"row {row_index} {field}")
