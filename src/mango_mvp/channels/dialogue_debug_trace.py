from __future__ import annotations

"""Optional JSONL trace for dialogue-contract diagnostics.

The logger is off by default and writes only when DIALOGUE_CONTRACT_DEBUG_TRACE=1
and a run directory/path is provided through the prompt context.
"""

import json
import os
import time
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping


DEBUG_TRACE_ENV = "DIALOGUE_CONTRACT_DEBUG_TRACE"
DEBUG_TRACE_FILE_NAME = "debug_trace.jsonl"
MAX_TRACE_FIELD_CHARS = 200
MAX_TRACE_ITEMS = 20
MAX_TRACE_DEPTH = 4
_TRACE_WRITE_LOCK = threading.Lock()


def trace_enabled(context: Mapping[str, Any] | None = None) -> bool:
    if isinstance(context, Mapping):
        cfg = context.get("dialogue_contract_debug_trace")
        if isinstance(cfg, Mapping) and "enabled" in cfg:
            return _truthy(cfg.get("enabled"))
        if "DIALOGUE_CONTRACT_DEBUG_TRACE" in context:
            return _truthy(context.get("DIALOGUE_CONTRACT_DEBUG_TRACE"))
    return _truthy(os.getenv(DEBUG_TRACE_ENV))


def trace_event(
    context: Mapping[str, Any] | None,
    node: str,
    values: Mapping[str, Any] | None = None,
    *,
    duration_ms: int | None = None,
) -> None:
    if not trace_enabled(context):
        return
    path = trace_path(context)
    if path is None:
        return
    payload = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "dialog_id": _trace_context_value(context, "dialog_id"),
        "turn": _trace_context_value(context, "turn"),
        "node": str(node or "")[:120],
        "values": _trim_value(dict(values or {})),
    }
    if duration_ms is not None:
        payload["duration_ms"] = max(0, int(duration_ms))
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with _TRACE_WRITE_LOCK:
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    except Exception:
        return


@contextmanager
def trace_span(
    context: Mapping[str, Any] | None,
    node: str,
    values: Mapping[str, Any] | None = None,
) -> Iterator[dict[str, Any]]:
    start = time.perf_counter()
    mutable = dict(values or {})
    try:
        yield mutable
    finally:
        trace_event(context, node, mutable, duration_ms=round((time.perf_counter() - start) * 1000))


def trace_path(context: Mapping[str, Any] | None = None) -> Path | None:
    if not isinstance(context, Mapping):
        return None
    cfg = context.get("dialogue_contract_debug_trace")
    if isinstance(cfg, Mapping):
        explicit_path = cfg.get("path")
        if explicit_path:
            return _safe_trace_path(Path(str(explicit_path)))
        run_dir = cfg.get("run_dir") or cfg.get("out_dir")
        if run_dir:
            return _safe_trace_path(Path(str(run_dir)) / DEBUG_TRACE_FILE_NAME)
    explicit_path = context.get("debug_trace_path")
    if explicit_path:
        return _safe_trace_path(Path(str(explicit_path)))
    run_dir = context.get("debug_trace_dir") or context.get("run_dir") or context.get("out_dir")
    if run_dir:
        return _safe_trace_path(Path(str(run_dir)) / DEBUG_TRACE_FILE_NAME)
    return None


def _safe_trace_path(path: Path) -> Path | None:
    resolved = path.expanduser().resolve(strict=False)
    if "stable_runtime" in resolved.parts:
        return None
    return resolved


def _trace_context_value(context: Mapping[str, Any] | None, key: str) -> Any:
    if not isinstance(context, Mapping):
        return "" if key == "dialog_id" else None
    cfg = context.get("dialogue_contract_debug_trace")
    if isinstance(cfg, Mapping) and key in cfg:
        return _trim_scalar(cfg.get(key))
    aliases = (key,)
    if key == "turn":
        aliases = ("turn", "turn_index", "turn_number")
    for alias in aliases:
        if alias in context:
            return _trim_scalar(context.get(alias))
    return "" if key == "dialog_id" else None


def _trim_value(value: Any, *, depth: int = 0) -> Any:
    if depth >= MAX_TRACE_DEPTH:
        return _trim_scalar(value)
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= MAX_TRACE_ITEMS:
                result["..."] = f"+{len(value) - MAX_TRACE_ITEMS} items"
                break
            result[str(key)[:80]] = _trim_value(item, depth=depth + 1)
        return result
    if isinstance(value, (list, tuple, set)):
        items = list(value)
        trimmed = [_trim_value(item, depth=depth + 1) for item in items[:MAX_TRACE_ITEMS]]
        if len(items) > MAX_TRACE_ITEMS:
            trimmed.append(f"+{len(items) - MAX_TRACE_ITEMS} items")
        return trimmed
    return _trim_scalar(value)


def _trim_scalar(value: Any) -> Any:
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    text = " ".join(str(value or "").split())
    if len(text) <= MAX_TRACE_FIELD_CHARS:
        return text
    return text[: MAX_TRACE_FIELD_CHARS - 1].rstrip() + "…"


def _truthy(value: Any) -> bool:
    return str(value or "").strip().casefold() in {"1", "true", "yes", "y", "on", "enabled"}
