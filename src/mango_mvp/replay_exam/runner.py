from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

from .machine_gate import run_machine_gate
from .models import BotReplayResult, ReplayCase, ReplayMessage


Provider = Callable[[ReplayCase, Mapping[str, object]], BotReplayResult]


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
        gate = run_machine_gate(case, result)
        rows.append(
            {
                "dialog_id": case.dialog_id,
                "turn_id": case.turn_id,
                "segment": case.segment,
                "route": result.route,
                "bot_text": result.bot_text,
                "safety_flags": list(result.safety_flags),
                "machine_gate": {"passed": gate.passed, "flags": list(gate.flags), "new_numbers": list(gate.new_numbers)},
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
