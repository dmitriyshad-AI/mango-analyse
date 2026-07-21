from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

from mango_mvp.channels.dialogue_memory import update_dialogue_memory_after_answer

from .machine_gate import number_index, run_machine_gate
from .models import BotReplayResult, ReplayCase, ReplayMessage
from .pseudonymizer import public_contact_allowlist


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
        "previous_turn",
        "raw",
        "blob",
    }
)
_PREFIX_MESSAGE_KEYS_V4 = frozenset({"from_me", "text", "ts_masked"})
_FORBIDDEN_CASE_KEYS = frozenset({"raw", "from", "to", "phone", "chatId", "contact_name", "username", "wappi_bot_id", "task_id", "stanzaId"})


def replay_memory_snapshot(memory: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    if not isinstance(memory, Mapping):
        memory = {}
    known_slots = memory.get("known_slots") if isinstance(memory.get("known_slots"), Mapping) else {}
    p0_latch = memory.get("p0_latch") if isinstance(memory.get("p0_latch"), Mapping) else {}
    do_not_reask = memory.get("do_not_reask_slots") or memory.get("do_not_ask_again") or ()
    return {
        "schema_version": "wappi_replay_memory_snapshot_v1",
        "known_slots": dict(known_slots),
        "do_not_reask": [str(item) for item in do_not_reask if str(item).strip()]
        if isinstance(do_not_reask, (list, tuple, set))
        else [],
        "p0_latch": _scrub_replay_p0_latch(p0_latch),
    }


def _scrub_replay_p0_latch(p0_latch: Mapping[str, Any]) -> dict[str, Any]:
    """Keep latch state useful for review without leaking runtime turn ids."""

    return {
        key: value
        for key, value in dict(p0_latch).items()
        if key not in {"trigger_turn_id", "release_event_id"}
    }


def _scrub_replay_memory_snapshot(snapshot: object) -> object:
    if not isinstance(snapshot, Mapping):
        return snapshot
    result = dict(snapshot)
    p0_latch = result.get("p0_latch")
    if isinstance(p0_latch, Mapping):
        result["p0_latch"] = _scrub_replay_p0_latch(p0_latch)
    return result


def sanitize_replay_row(row: Mapping[str, object]) -> dict[str, object]:
    result = dict(row)
    for key in ("memory_snapshot", "memory_snapshot_after"):
        result[key] = _scrub_replay_memory_snapshot(result.get(key))
    return result


def sanitize_replay_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    return [sanitize_replay_row(row) for row in rows]


def _brand_from_payload(obj: Mapping[str, Any]) -> str:
    raw = str(obj.get("brand") or "").strip().casefold()
    if raw:
        return raw
    contour = str(obj.get("contour") or "").strip().casefold()
    if "foton" in contour or "фотон" in contour:
        return "foton"
    if "unpk" in contour or "унпк" in contour or "лвш" in contour:
        return "unpk"
    raise ValueError("replay case must include brand or a contour that resolves to foton/unpk")


def _validate_case_payload(obj: Mapping[str, Any], *, line_no: int) -> None:
    forbidden = sorted(key for key in obj if str(key) in _FORBIDDEN_CASE_KEYS)
    if forbidden:
        raise ValueError(f"unsafe replay case keys at line {line_no}: {forbidden}")
    for index, item in enumerate(obj.get("prefix_messages") or ()):
        if not isinstance(item, Mapping):
            raise ValueError(f"prefix_messages[{index}] at line {line_no} must be an object")
        keys = {str(key) for key in item}
        if keys != _PREFIX_MESSAGE_KEYS_V4:
            raise ValueError(
                f"prefix_messages[{index}] at line {line_no} must use exact keys "
                f"{sorted(_PREFIX_MESSAGE_KEYS_V4)}, got {sorted(keys)}"
            )
        if not isinstance(item.get("from_me"), bool):
            raise ValueError(f"prefix_messages[{index}].from_me at line {line_no} must be boolean")
        if not isinstance(item.get("text"), str):
            raise ValueError(f"prefix_messages[{index}].text at line {line_no} must be string")
        if not isinstance(item.get("ts_masked"), str):
            raise ValueError(f"prefix_messages[{index}].ts_masked at line {line_no} must be string")


def _turn_index_from_payload(obj: Mapping[str, Any]) -> int:
    raw = obj.get("turn_index")
    if raw is not None:
        try:
            return int(raw)
        except (TypeError, ValueError):
            return 0
    turn_id = str(obj.get("turn_id") or obj.get("exam_id") or "")
    match = re.search(r"#(\d+)$", turn_id)
    return int(match.group(1)) if match else 0


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
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        obj = json.loads(line)
        if not isinstance(obj, Mapping):
            raise ValueError(f"replay case at line {line_no} must be an object")
        _validate_case_payload(obj, line_no=line_no)
        dialog_id = str(obj.get("dialog_id") or obj.get("dialog_key_masked") or "")
        turn_index = _turn_index_from_payload(obj)
        turn_id = str(obj.get("turn_id") or obj.get("exam_id") or (f"{dialog_id}#{turn_index}" if dialog_id else ""))
        prefix_items = [item for item in (obj.get("prefix_messages") or ()) if isinstance(item, Mapping)]
        brand = _brand_from_payload(obj)
        prefix_messages = tuple(
            ReplayMessage(
                profile_id=str(obj.get("profile_id") or ""),
                chat_id=str(obj.get("chat_id") or ""),
                message_id="",
                text=str(item.get("text") or ""),
                timestamp=int(item.get("timestamp") or index + 1),
                from_me=bool(item["from_me"]),
                ts_masked=str(item.get("ts_masked") or ""),
                sender_name="",
                raw={},
            )
            for index, item in enumerate(prefix_items)
        )
        metadata = dict(obj.get("meta") or obj.get("metadata") or {})
        cases.append(
            ReplayCase(
                dialog_id=dialog_id,
                profile_id=str(obj.get("profile_id") or ""),
                chat_id=str(obj.get("chat_id") or ""),
                turn_id=turn_id,
                brand=brand,
                client_message=str(obj["client_message"]),
                manager_reference=str(obj.get("manager_reference") or ""),
                turn_index=turn_index,
                contour=str(obj.get("contour") or brand),
                dialog_key_masked=str(obj.get("dialog_key_masked") or dialog_id),
                prefix_messages=prefix_messages,
                segment=str(obj.get("segment") or "chat_only"),
                expected_p0=bool(obj.get("expected_p0")),
                metadata=metadata,
            )
        )
    return cases


def _run_dialog_cases(dialog_cases: Sequence[ReplayCase], provider: Provider) -> list[dict[str, object]]:
    older_summary = ""
    dialogue_memory: Mapping[str, Any] = {}
    rows: list[dict[str, object]] = []
    for case in dialog_cases:
        memory_before = replay_memory_snapshot(dialogue_memory)
        result = provider(
            case,
            {
                "older_summary": older_summary,
                "dialogue_memory": dialogue_memory,
                "sends_client_replies": False,
                "crm_context": {},
            },
        )
        client_safe_fact_texts = _current_turn_client_safe_fact_texts(result.metadata)
        client_safe_numbers = _current_turn_client_safe_numbers(result.metadata)
        gate = run_machine_gate(
            case,
            result,
            client_safe_numbers=client_safe_numbers,
            pii_allowlist=public_contact_allowlist(client_safe_fact_texts),
        )
        next_memory = update_dialogue_memory_after_answer(
            dialogue_memory,
            answer_text=result.bot_text,
            route=result.route,
            safety_flags=result.safety_flags,
            memory_llm_fn=None,
            context={"replay_exam": True, "turn_id": case.turn_id},
        ).to_json_dict()
        rows.append(
            {
                "dialog_id": case.dialog_id,
                "turn_id": case.turn_id,
                "turn_index": case.turn_index,
                "segment": case.segment,
                "route": result.route,
                "bot_text": result.bot_text,
                "safety_flags": list(result.safety_flags),
                "provider_metadata": dict(result.metadata),
                "memory_snapshot": memory_before,
                "memory_snapshot_after": replay_memory_snapshot(next_memory),
                "machine_gate": {
                    "passed": gate.passed,
                    "flags": list(gate.flags),
                    "new_numbers": list(gate.new_numbers),
                    "client_safe_numbers_count": len(client_safe_numbers),
                },
                "llm_calls_client": 0,
            }
        )
        dialogue_memory = next_memory
        older_summary = (older_summary + "\n" + f"client: {case.client_message}\nbot: {result.bot_text}").strip()[-4000:]
    return rows


def run_replay_exam(
    cases: Iterable[ReplayCase],
    provider: Provider,
    *,
    parallel_dialogs: int = 4,
    progress_callback: Callable[[Sequence[Mapping[str, object]]], None] | None = None,
) -> list[dict[str, object]]:
    by_dialog: dict[str, list[ReplayCase]] = {}
    for case in cases:
        by_dialog.setdefault(case.dialog_id, []).append(case)
    for dialog_cases in by_dialog.values():
        dialog_cases.sort(key=lambda item: (item.turn_index, item.turn_id))
    rows: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=max(1, parallel_dialogs)) as pool:
        futures = [pool.submit(_run_dialog_cases, tuple(dialog_cases), provider) for dialog_cases in by_dialog.values()]
        for future in as_completed(futures):
            dialog_rows = future.result()
            rows.extend(dialog_rows)
            if progress_callback is not None:
                progress_callback(tuple(dialog_rows))
    rows.sort(key=lambda item: (str(item["dialog_id"]), int(item.get("turn_index") or 0), str(item["turn_id"])))
    return rows


def write_replay_outputs(out_dir: Path, rows: Sequence[Mapping[str, object]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    clean_rows = sanitize_replay_rows(rows)
    (out_dir / "replay_results.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in clean_rows),
        encoding="utf-8",
    )
    summary = {
        "schema_version": "wappi_replay_exam_summary_v1",
        "metric": "chat_only_replay",
        "turns": len(clean_rows),
        "machine_gate_failures": sum(1 for row in clean_rows if not (row.get("machine_gate") or {}).get("passed")),
        "llm_calls": {"client": sum(int(row.get("llm_calls_client") or 0) for row in clean_rows)},
    }
    (out_dir / "replay_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
