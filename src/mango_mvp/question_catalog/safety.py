from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

from mango_mvp.insights.sanitizers import (
    has_any_safety_risk,
    sanitize_answer,
)


EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d[\s()\-]*){10,15}(?!\d)")


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
    text = sanitized.text.strip()
    if source and not text:
        return "[очищено: исходный текст полностью состоял из небезопасных данных]", tuple(
            dict.fromkeys((*sanitized.flags, "empty_after_public_redaction"))
        )
    if max_chars > 0 and len(text) > max_chars:
        text = text[: max_chars - 1].rstrip() + "..."
    if has_any_safety_risk(text) or contains_raw_contact(text):
        return "[очищено: потенциальные персональные или изменяемые данные]", tuple(
            dict.fromkeys((*sanitized.flags, "blocked_unresolved_public_risk"))
        )
    return text, tuple(sanitized.flags)


def contains_raw_contact(value: Any) -> bool:
    text = str(value or "")
    return bool(EMAIL_RE.search(text) or PHONE_RE.search(text))


def assert_public_text_safe(value: Any, *, field_name: str = "text") -> None:
    text = str(value or "")
    if contains_raw_contact(text) or has_any_safety_risk(text):
        raise ValueError(f"{field_name} contains unsafe public text")


def assert_public_rows_safe(rows: Iterable[dict[str, Any]], *, text_fields: Iterable[str]) -> None:
    fields = tuple(text_fields)
    for row_index, row in enumerate(rows, start=1):
        for field in fields:
            assert_public_text_safe(row.get(field, ""), field_name=f"row {row_index} {field}")
