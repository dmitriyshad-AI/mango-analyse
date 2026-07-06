from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .machine_gate import number_index, run_machine_gate
from .models import BotReplayResult, ReplayCase, ReplayMessage


Provider = Callable[[ReplayCase, Mapping[str, object]], BotReplayResult]

_CURRENT_TURN_FACT_CONTAINERS = frozenset(
    {
        "retrieved_facts",
        "client_safe_facts",
        "bot_confirmed_facts",
        "confirmed_facts_for_judge",
    }
)
_CURRENT_TURN_FACT_ROOTS = frozenset({"direct_path", "dialogue_contract_pipeline"})
_CLIENT_SAFE_TEXT_KEYS = frozenset({"client_safe_text", "client_text", "safe_text", "text"})
_BLOCKED_FACT_TEXT_KEYS = frozenset({"internal_only_text", "manager_only_text", "manager_text"})
_BLOCKED_METADATA_KEYS = frozenset(
    {
        "manager_reference",
        "raw_response",
        "replay_raw_response",
        "older_summary",
        "history",
        "recent_messages",
        "prefix_messages",
        "raw",
        "blob",
    }
)


def _fact_payload_texts(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Mapping):
        texts: list[str] = []
        for key, nested in value.items():
            key_s = str(key)
            key_l = key_s.casefold()
            if key_l in _BLOCKED_FACT_TEXT_KEYS:
                continue
            if key_s in _CLIENT_SAFE_TEXT_KEYS and isinstance(nested, str):
                texts.append(nested)
            elif key_s in _BLOCKED_METADATA_KEYS:
                continue
            elif isinstance(nested, str):
                texts.append(nested)
            elif isinstance(nested, (Mapping, list, tuple)):
                texts.extend(_fact_payload_texts(nested))
        return tuple(texts)
    if isinstance(value, (list, tuple)):
        texts: list[str] = []
        for item in value:
            texts.extend(_fact_payload_texts(item))
        return tuple(texts)
    return ()


def _current_turn_client_safe_fact_texts(metadata: Mapping[str, Any]) -> tuple[str, ...]:
    texts: list[str] = []

    def walk(value: object, *, key: str = "", inside_current_root: bool = False) -> None:
        key_s = key.casefold()
        if key_s in _BLOCKED_METADATA_KEYS:
            return
        next_inside_current_root = inside_current_root or key_s in _CURRENT_TURN_FACT_ROOTS
        if next_inside_current_root and key_s in _CURRENT_TURN_FACT_CONTAINERS:
            texts.extend(_fact_payload_texts(value))
            return
        if isinstance(value, Mapping):
            for nested_key, nested_value in value.items():
                walk(nested_value, key=str(nested_key), inside_current_root=next_inside_current_root)
        elif isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, (Mapping, list, tuple)):
                    walk(item, inside_current_root=next_inside_current_root)

    walk(metadata)
    return tuple(text for text in texts if str(text or "").strip())


def _current_turn_client_safe_numbers(metadata: Mapping[str, Any]) -> tuple[str, ...]:
    texts = _current_turn_client_safe_fact_texts(metadata)
    numbers = set(number_index(texts))
    for text in texts:
        numbers.update(re.findall(r"\d+", str(text or "")))
    return tuple(sorted(numbers))


def load_cases(path: Path) -> list[ReplayCase]:
    cases: list[ReplayCase] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        prefix_messages = tuple(
            ReplayMessage(
                profile_id=str(item.get("profile_id") or ""),
                chat_id=str(item.get("chat_id") or ""),
                message_id=str(item.get("message_id") or ""),
                text=str(item.get("text") or ""),
                timestamp=int(item.get("timestamp") or 0),
                from_me=bool(item.get("from_me")),
                sender_name=str(item.get("sender_name") or ""),
                raw=dict(item.get("raw") or {}),
            )
            for item in (obj.get("prefix_messages") or ())
            if isinstance(item, Mapping)
        )
        cases.append(
            ReplayCase(
                dialog_id=str(obj["dialog_id"]),
                profile_id=str(obj.get("profile_id") or ""),
                chat_id=str(obj.get("chat_id") or ""),
                turn_id=str(obj["turn_id"]),
                brand=str(obj["brand"]),
                client_message=str(obj["client_message"]),
                manager_reference=str(obj.get("manager_reference") or ""),
                prefix_messages=prefix_messages,
                segment=str(obj.get("segment") or "chat_only"),
                expected_p0=bool(obj.get("expected_p0")),
                metadata=dict(obj.get("metadata") or {}),
            )
        )
    return cases


def _run_dialog_cases(dialog_cases: Sequence[ReplayCase], provider: Provider) -> list[dict[str, object]]:
    older_summary = ""
    rows: list[dict[str, object]] = []
    for case in dialog_cases:
        result = provider(case, {"older_summary": older_summary, "sends_client_replies": False, "crm_context": {}})
        client_safe_numbers = _current_turn_client_safe_numbers(result.metadata)
        gate = run_machine_gate(case, result, client_safe_numbers=client_safe_numbers)
        rows.append(
            {
                "dialog_id": case.dialog_id,
                "turn_id": case.turn_id,
                "segment": case.segment,
                "route": result.route,
                "bot_text": result.bot_text,
                "safety_flags": list(result.safety_flags),
                "provider_metadata": dict(result.metadata),
                "machine_gate": {
                    "passed": gate.passed,
                    "flags": list(gate.flags),
                    "new_numbers": list(gate.new_numbers),
                    "client_safe_numbers_count": len(client_safe_numbers),
                },
                "llm_calls_client": 0,
            }
        )
        older_summary = (older_summary + "\n" + f"client: {case.client_message}\nbot: {result.bot_text}").strip()[-4000:]
    return rows


def run_replay_exam(cases: Iterable[ReplayCase], provider: Provider, *, parallel_dialogs: int = 4) -> list[dict[str, object]]:
    by_dialog: dict[str, list[ReplayCase]] = {}
    for case in cases:
        by_dialog.setdefault(case.dialog_id, []).append(case)
    for dialog_cases in by_dialog.values():
        dialog_cases.sort(key=lambda item: item.turn_id)
    rows: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=max(1, parallel_dialogs)) as pool:
        futures = [pool.submit(_run_dialog_cases, tuple(dialog_cases), provider) for dialog_cases in by_dialog.values()]
        for future in as_completed(futures):
            rows.extend(future.result())
    rows.sort(key=lambda item: (str(item["dialog_id"]), str(item["turn_id"])))
    return rows


def write_replay_outputs(out_dir: Path, rows: Sequence[Mapping[str, object]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "replay_results.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    summary = {
        "schema_version": "wappi_replay_exam_summary_v1",
        "metric": "chat_only_replay",
        "turns": len(rows),
        "machine_gate_failures": sum(1 for row in rows if not (row.get("machine_gate") or {}).get("passed")),
        "llm_calls": {"client": sum(int(row.get("llm_calls_client") or 0) for row in rows)},
    }
    (out_dir / "replay_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
