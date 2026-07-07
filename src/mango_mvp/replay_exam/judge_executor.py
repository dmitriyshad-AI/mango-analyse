from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from mango_mvp.channels.subscription_llm_parts.codex_exec import (
    CodexExecConfig,
    build_codex_exec_env,
    extract_json_object,
)

from .judge import PairedJudgePayload, build_balanced_replay_judge_payloads
from .models import BotReplayResult, ReplayCase


JudgeRunner = Callable[[Mapping[str, object]], Mapping[str, object]]


@dataclass(frozen=True)
class ReplayJudgeRequest:
    exam_id: str
    payload: Mapping[str, object]
    hidden_key: Mapping[str, str]


def _row_exam_id(row: Mapping[str, Any]) -> str:
    return str(row.get("turn_id") or row.get("exam_id") or "")


def _row_machine_gate_passed(row: Mapping[str, Any]) -> bool:
    gate = row.get("machine_gate")
    return isinstance(gate, Mapping) and bool(gate.get("passed"))


def _candidate_result(row: Mapping[str, Any]) -> BotReplayResult:
    flags = tuple(str(item) for item in (row.get("safety_flags") or ()) if str(item).strip())
    metadata = row.get("provider_metadata") if isinstance(row.get("provider_metadata"), Mapping) else {}
    return BotReplayResult(
        route=str(row.get("route") or "draft_for_manager"),
        bot_text=str(row.get("bot_text") or ""),
        safety_flags=flags,
        metadata=dict(metadata),
    )


def _baseline_result(case: ReplayCase) -> BotReplayResult:
    return BotReplayResult(route="manager_reference", bot_text=case.manager_reference)


def build_replay_judge_requests(
    cases: Sequence[ReplayCase],
    replay_rows: Sequence[Mapping[str, Any]],
    *,
    seed: str = "replay_judge_v1",
    max_judge_calls: Optional[int] = None,
) -> list[ReplayJudgeRequest]:
    if max_judge_calls is not None and max_judge_calls < 1:
        raise ValueError("max_judge_calls must be positive")
    rows_by_exam_id = {_row_exam_id(row): row for row in replay_rows if _row_exam_id(row)}
    triples: list[tuple[ReplayCase, BotReplayResult, BotReplayResult]] = []
    for case in cases:
        row = rows_by_exam_id.get(case.turn_id)
        if row is None:
            continue
        if case.segment != "chat_only":
            continue
        if not case.manager_reference.strip():
            continue
        if not _row_machine_gate_passed(row):
            continue
        candidate = _candidate_result(row)
        if not candidate.bot_text.strip():
            continue
        triples.append((case, _baseline_result(case), candidate))
    payloads: list[PairedJudgePayload] = build_balanced_replay_judge_payloads(triples, seed=seed)
    if max_judge_calls is not None:
        payloads = payloads[:max_judge_calls]
    return [
        ReplayJudgeRequest(
            exam_id=str(payload.payload.get("exam_id") or ""),
            payload=payload.payload,
            hidden_key=payload.hidden_key,
        )
        for payload in payloads
    ]


def write_replay_judge_payloads(out_dir: Path, requests: Sequence[ReplayJudgeRequest]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "judge_payloads.jsonl").write_text(
        "".join(json.dumps(dict(request.payload), ensure_ascii=False, sort_keys=True) + "\n" for request in requests),
        encoding="utf-8",
    )
    (out_dir / "judge_key.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "schema_version": "replay_judge_hidden_key_v1",
                    "exam_id": request.exam_id,
                    "hidden_key": dict(request.hidden_key),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
            for request in requests
        ),
        encoding="utf-8",
    )


def _judge_prompt(payload: Mapping[str, object]) -> str:
    return (
        "Ты независимый судья качества ответа учебного центра. "
        "Сравни answer_a и answer_b по вопросу клиента и контексту. "
        "Не угадывай скрытый ключ. Верни только JSON: "
        '{"winner":"answer_a|answer_b|tie","bot_send_as_is":true|false,'
        '"flags":["..."],"reason":"кратко"}.\n\n'
        f"PAYLOAD:\n{json.dumps(dict(payload), ensure_ascii=False, sort_keys=True)}"
    )


class CodexReplayJudgeRunner:
    def __init__(
        self,
        *,
        config: Optional[CodexExecConfig] = None,
        cache_dir: Path = Path("~/.mango_local/replay_exam/judge_cache").expanduser(),
    ) -> None:
        self.config = config or CodexExecConfig()
        self.cache_dir = cache_dir.expanduser()

    def __call__(self, payload: Mapping[str, object]) -> Mapping[str, object]:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="replay_judge_", dir=str(self.cache_dir)) as tmp_dir:
            output_path = Path(tmp_dir) / "judge_response.txt"
            cmd = self.config.build_command(output_path)
            completed = subprocess.run(
                cmd,
                input=_judge_prompt(payload),
                text=True,
                capture_output=True,
                env=build_codex_exec_env(),
                timeout=180,
                check=False,
            )
            if completed.returncode != 0:
                raise RuntimeError(f"replay judge failed: {completed.stderr.strip()[:500]}")
            raw = output_path.read_text(encoding="utf-8", errors="replace") if output_path.exists() else completed.stdout
            return extract_json_object(raw)


def _normalize_judge_result(raw: Mapping[str, object]) -> dict[str, object]:
    winner = str(raw.get("winner") or "tie").strip().casefold()
    if winner not in {"answer_a", "answer_b", "tie"}:
        winner = "tie"
    raw_flags = raw.get("flags") or ()
    if isinstance(raw_flags, str):
        raw_flags = [raw_flags]
    flags = [str(item).strip() for item in raw_flags if str(item).strip()]
    return {
        "winner": winner,
        "bot_send_as_is": bool(raw.get("bot_send_as_is")),
        "flags": flags[:12],
        "reason": str(raw.get("reason") or "").strip()[:800],
    }


def execute_replay_judge_requests(
    out_dir: Path,
    requests: Sequence[ReplayJudgeRequest],
    *,
    runner: JudgeRunner,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for request in requests:
        raw = runner(request.payload)
        rows.append(
            {
                "schema_version": "replay_judge_result_v1",
                "judge_version": "replay_judge_v1",
                "exam_id": request.exam_id,
                "result": _normalize_judge_result(raw),
            }
        )
    (out_dir / "judge_results.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    summary = {
        "schema_version": "replay_judge_summary_v1",
        "judge_version": "replay_judge_v1",
        "calls": len(rows),
        "bot_send_as_is": sum(1 for row in rows if (row.get("result") or {}).get("bot_send_as_is")),
        "winner_counts": {
            label: sum(1 for row in rows if (row.get("result") or {}).get("winner") == label)
            for label in ("answer_a", "answer_b", "tie")
        },
    }
    (out_dir / "judge_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return rows
