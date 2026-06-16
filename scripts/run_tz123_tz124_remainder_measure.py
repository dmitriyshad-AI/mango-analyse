#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections import Counter
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from mango_mvp.channels.dialogue_memory import MEMORY_PROVENANCE_ENV, build_dialogue_memory
from mango_mvp.channels.new_lead_funnel import ANCHORED_BARE_GRADE_ENV
from mango_mvp.channels.subscription_llm import SubscriptionDraftResult
from mango_mvp.channels.subscription_llm_parts.post_layers import (
    apply_question_instead_of_handoff_layer,
)
from mango_mvp.channels.subscription_llm_parts.support import (
    DIRECT_PATH_ENV,
    DIRECT_PATH_PILOT_CONFIG_ENV,
    DIRECT_PATH_PILOT_CONFIG_VERSION,
    QUESTION_INSTEAD_OF_HANDOFF_ENV,
)


DEFAULT_HISTORY_ROOT = Path("/Users/dmitrijfabarisov/Projects/Mango analyse/runs")
DEFAULT_SNAPSHOT = Path("product_data/knowledge_base/kb_release_20260612_v6_7_staging_r4_1/kb_release_v3_snapshot.json")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    with tz124_env(question_enabled=True):
        initial_rows = build_layer_candidates(args.history_root.expanduser())
    write_jsonl(out_dir / "initial_layer_candidates.jsonl", initial_rows)
    write_replay_pack(
        initial_rows,
        scenarios_path=out_dir / "initial_scenarios.jsonl",
        replay_path=out_dir / "initial_replay_source.jsonl",
        title="TZ123+TZ124 initial C0 candidates",
    )

    summary: dict[str, Any] = {
        "schema_version": "tz123_tz124_remainder_measure_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "history_root": str(args.history_root),
        "out_dir": str(out_dir),
        "snapshot": str(args.snapshot),
        "initial_candidates": len(initial_rows),
        "initial_by_slot": dict(Counter(str(row.get("slot") or "") for row in initial_rows)),
        "llm_calls_total": 0,
        "safety": {
            "writes_crm": False,
            "writes_tallanto": False,
            "writes_amo": False,
            "sends_messages": False,
            "runs_asr": False,
            "touches_stable_runtime": False,
        },
    }

    if not args.run_replay or not initial_rows:
        summary["replay_ran"] = False
        (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    off_dir = out_dir / "off_replay"
    on_dir = out_dir / "on_replay"
    run_dynamic_replay(
        scenarios=out_dir / "initial_scenarios.jsonl",
        replay_source=out_dir / "initial_replay_source.jsonl",
        out_dir=off_dir,
        snapshot=args.snapshot,
        parallel=args.parallel,
        question_enabled=False,
        timeout_sec=args.timeout_sec,
        model=args.model,
    )

    with tz124_env(question_enabled=True):
        remainder_rows = build_layer_candidates_from_transcripts(off_dir / "dynamic_dialog_transcripts.jsonl")
    write_jsonl(out_dir / "remainder_rows.jsonl", remainder_rows)
    write_replay_pack(
        remainder_rows,
        scenarios_path=out_dir / "remainder_scenarios.jsonl",
        replay_path=out_dir / "remainder_replay_source.jsonl",
        title="TZ123+TZ124 remainder after current OFF replay",
    )
    summary["off_replay"] = summarize_dynamic_run(off_dir)
    summary["remainder_candidates"] = len(remainder_rows)
    summary["remainder_by_slot"] = dict(Counter(str(row.get("slot") or "") for row in remainder_rows))

    if remainder_rows:
        run_dynamic_replay(
            scenarios=out_dir / "remainder_scenarios.jsonl",
            replay_source=out_dir / "remainder_replay_source.jsonl",
            out_dir=on_dir,
            snapshot=args.snapshot,
            parallel=args.parallel,
            question_enabled=True,
            timeout_sec=args.timeout_sec,
            model=args.model,
        )
        summary["on_replay"] = summarize_dynamic_run(on_dir)
        summary["semantic_review"] = semantic_review(on_dir)
    else:
        summary["on_replay"] = {"ran": False}
        summary["semantic_review"] = {"verdict": "BLOCKED", "reason": "empty_remainder"}

    summary["llm_calls_total"] = (
        int((summary.get("off_replay") or {}).get("llm_calls_total") or 0)
        + int((summary.get("on_replay") or {}).get("llm_calls_total") or 0)
    )
    summary["replay_ran"] = True
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure TZ123 question layer on the TZ124 remainder.")
    parser.add_argument("--history-root", type=Path, default=DEFAULT_HISTORY_ROOT)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--out-dir", type=Path, default=Path("audits/_inbox/tz123_tz124_remainder_20260616"))
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--timeout-sec", type=int, default=180)
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--run-replay", action="store_true")
    args = parser.parse_args(argv)
    if args.parallel < 1:
        raise SystemExit("--parallel must be >= 1")
    return args


@contextmanager
def tz124_env(*, question_enabled: bool) -> Iterable[None]:
    keys = (
        ANCHORED_BARE_GRADE_ENV,
        MEMORY_PROVENANCE_ENV,
        DIRECT_PATH_PILOT_CONFIG_ENV,
        QUESTION_INSTEAD_OF_HANDOFF_ENV,
    )
    previous = {key: os.environ.get(key) for key in keys}
    os.environ[ANCHORED_BARE_GRADE_ENV] = "1"
    os.environ[MEMORY_PROVENANCE_ENV] = "1"
    os.environ[DIRECT_PATH_PILOT_CONFIG_ENV] = DIRECT_PATH_PILOT_CONFIG_VERSION
    os.environ[QUESTION_INSTEAD_OF_HANDOFF_ENV] = "1" if question_enabled else "0"
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def build_layer_candidates(history_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, int, str]] = set()
    for path in sorted(history_root.glob("**/dynamic_dialog_transcripts.jsonl")):
        for dialog in read_jsonl(path):
            if not isinstance(dialog, Mapping):
                continue
            for turn in dialog.get("turns") or []:
                if not isinstance(turn, Mapping) or not is_answer_only_handoff(turn):
                    continue
                row = layer_candidate_from_turn(dialog, turn, source_path=path)
                if not row:
                    continue
                key = (
                    str(row.get("brand") or ""),
                    int(row.get("source_turn") or 0),
                    normalize_message(str(row.get("client_message") or "")),
                )
                if key in seen:
                    continue
                seen.add(key)
                rows.append(row)
    rows.sort(key=lambda row: (str(row.get("brand") or ""), str(row.get("slot") or ""), str(row.get("dialog_id") or "")))
    return rows


def build_layer_candidates_from_transcripts(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dialog in read_jsonl(path):
        if not isinstance(dialog, Mapping):
            continue
        for turn in dialog.get("turns") or []:
            if not isinstance(turn, Mapping) or not is_answer_only_handoff(turn):
                continue
            row = layer_candidate_from_turn(dialog, turn, source_path=path)
            if row:
                rows.append(row)
    rows.sort(key=lambda row: str(row.get("dialog_id") or ""))
    return rows


def layer_candidate_from_turn(dialog: Mapping[str, Any], turn: Mapping[str, Any], *, source_path: Path) -> dict[str, Any] | None:
    brand = str(dialog.get("brand") or (dialog.get("persona") or {}).get("brand") or "unknown").strip().lower()
    if brand not in {"unpk", "foton"}:
        return None
    client_message = str(turn.get("client_message") or "").strip()
    if not client_message:
        return None
    result = result_from_turn(turn)
    context = context_from_turn(dialog, turn, source_path=source_path)
    candidate = apply_question_instead_of_handoff_layer(result, client_message=client_message, context=context)
    meta = candidate.metadata.get("question_instead_of_handoff") if isinstance(candidate.metadata, Mapping) else {}
    if not isinstance(meta, Mapping) or meta.get("status") != "fired":
        return None
    slot = str(meta.get("slot") or "").strip()
    if not slot:
        return None
    return {
        "dialog_id": stable_dialog_id(source_path, dialog, turn),
        "brand": brand,
        "slot": slot,
        "question": candidate.draft_text,
        "client_message": client_message,
        "source_run": source_path.parent.name,
        "source_path": str(source_path),
        "source_dialog_id": str(dialog.get("dialog_id") or ""),
        "source_turn": int(turn.get("turn") or 1),
        "source_route": str(turn.get("bot_route") or ""),
        "source_action": str((turn.get("bot_action_decision") or {}).get("action") or ""),
        "source_reason": str((turn.get("bot_action_decision") or {}).get("reason") or ""),
    }


def result_from_turn(turn: Mapping[str, Any]) -> SubscriptionDraftResult:
    metadata = {
        "authoritative_output_gate": dict(turn.get("bot_authoritative_output_gate") or {}),
        "action_decision": dict(turn.get("bot_action_decision") or {}),
        "action_proposal": dict(turn.get("bot_action_proposal") or {}),
        "direct_path": dict(turn.get("bot_direct_path") or {}),
        "answer_contract": dict(turn.get("bot_answer_contract") or {}),
    }
    return SubscriptionDraftResult(
        message_type=str(turn.get("bot_message_type") or "question"),
        topic_id=str(turn.get("bot_topic_id") or "service:S2_unclear"),
        risk_level=str(turn.get("bot_risk_level") or "low"),
        route=str(turn.get("bot_route") or "manager_only"),
        draft_text=str(turn.get("bot_text") or ""),
        manager_checklist=tuple(str(item) for item in (turn.get("bot_manager_checklist") or [])),
        missing_facts=tuple(str(item) for item in (turn.get("bot_missing_facts") or [])),
        safety_flags=tuple(str(item) for item in (turn.get("bot_safety_flags") or [])),
        metadata=metadata,
    )


def context_from_turn(dialog: Mapping[str, Any], turn: Mapping[str, Any], *, source_path: Path) -> dict[str, Any]:
    brand = str(dialog.get("brand") or (dialog.get("persona") or {}).get("brand") or "unknown").strip().lower()
    client_message = str(turn.get("client_message") or "")
    memory = build_dialogue_memory(
        current_message=client_message,
        active_brand=brand,
        recent_messages=(f"Клиент: {client_message}",),
        known_slots={},
        session_id=f"tz123_tz124:{stable_dialog_id(source_path, dialog, turn)}",
    ).to_prompt_view()
    answer_contract = turn.get("bot_answer_contract") if isinstance(turn.get("bot_answer_contract"), Mapping) else {}
    known_slots = answer_contract.get("known_slots") if isinstance(answer_contract.get("known_slots"), Mapping) else {}
    return {
        "active_brand": brand,
        DIRECT_PATH_ENV: "1",
        DIRECT_PATH_PILOT_CONFIG_ENV: DIRECT_PATH_PILOT_CONFIG_VERSION,
        QUESTION_INSTEAD_OF_HANDOFF_ENV: "1",
        "known_slots": dict(known_slots),
        "known_dialog_fields": dict(memory.get("known_slots") or {}),
        "dialogue_memory_view": memory,
        "answer_contract": dict(answer_contract),
        "conversation_intent_plan": dict(turn.get("bot_conversation_intent_plan") or {}),
        "recent_messages": [f"Клиент: {client_message}"],
    }


def is_answer_only_handoff(turn: Mapping[str, Any]) -> bool:
    decision = turn.get("bot_action_decision")
    return (
        str(turn.get("bot_route") or "") == "draft_for_manager"
        and isinstance(decision, Mapping)
        and str(decision.get("action") or "") == "answer_only"
        and not bool(decision.get("p0_latched"))
    )


def write_replay_pack(rows: Sequence[Mapping[str, Any]], *, scenarios_path: Path, replay_path: Path, title: str) -> None:
    scenarios: list[Mapping[str, Any]] = [
        {
            "type": "simulator_spec",
            "title": title,
            "rules": [
                "Replay-only set from current TZ123+TZ124 remainder.",
                "Acceptance is by question_instead_of_handoff counters and semantic grep.",
            ],
        },
        {
            "type": "judge_spec",
            "title": f"{title} fake judge",
            "instructions": "Judge is fake; acceptance is by deterministic transcript checks.",
            "hard_gates_fail_whole_dialog_if_violated": {
                "p0_mishandled": "P0 must not be converted into a clarification question",
                "brand_leak": "answer must remain within active brand",
            },
            "soft_flags": {},
        },
    ]
    replay_rows: list[Mapping[str, Any]] = []
    for row in rows:
        dialog_id = str(row.get("dialog_id") or "")
        brand = str(row.get("brand") or "unknown")
        scenarios.append(
            {
                "type": "persona",
                "dialog_id": dialog_id,
                "brand": brand,
                "category": "tz123_tz124_remainder",
                "persona": f"remainder: missing {row.get('slot')}",
                "mood": "деловой",
                "style": "коротко",
                "goal": "получить точный факт без лишнего ухода к менеджеру",
                "held_facts": {},
                "behaviors": [
                    f"Source: {row.get('source_run')} / {row.get('source_dialog_id')} / turn {row.get('source_turn')}"
                ],
                "max_turns": 1,
                "expected_route": "bot_answer_self",
                "success_criteria": "При ON задаётся один короткий уточняющий вопрос; P0 и бренд не нарушены.",
                "fail_criteria": "P0/бренд-регрессия, повторный допрос, цикл или уход к менеджеру без вопроса.",
            }
        )
        replay_rows.append(
            {
                "dialog_id": dialog_id,
                "brand": brand,
                "source_run": row.get("source_run"),
                "source_dialog_id": row.get("source_dialog_id"),
                "source_turn": row.get("source_turn"),
                "turns": [
                    {
                        "turn": 1,
                        "client_message": row.get("client_message"),
                        "client_stop": True,
                    }
                ],
            }
        )
    write_jsonl(scenarios_path, scenarios)
    write_jsonl(replay_path, replay_rows)


def run_dynamic_replay(
    *,
    scenarios: Path,
    replay_source: Path,
    out_dir: Path,
    snapshot: Path,
    parallel: int,
    question_enabled: bool,
    timeout_sec: int,
    model: str,
) -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = "src"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env[DIRECT_PATH_PILOT_CONFIG_ENV] = DIRECT_PATH_PILOT_CONFIG_VERSION
    env[ANCHORED_BARE_GRADE_ENV] = "1"
    env[MEMORY_PROVENANCE_ENV] = "1"
    env[QUESTION_INSTEAD_OF_HANDOFF_ENV] = "1" if question_enabled else "0"
    cmd = [
        sys.executable,
        "scripts/run_telegram_dynamic_client_sim.py",
        "--scenarios",
        str(scenarios),
        "--snapshot",
        str(snapshot),
        "--replay-from",
        str(replay_source),
        "--out-dir",
        str(out_dir),
        "--parallel",
        str(parallel),
        "--max-turns",
        "1",
        "--bot-mode",
        "codex",
        "--judge-mode",
        "fake",
        "--client-mode",
        "fake",
        "--memory-mode",
        "off",
        "--semantic-mode",
        "fake",
        "--semantic-verifier-mode",
        "fake",
        "--judge-prompt-version",
        "v9.1",
        "--timeout-sec",
        str(timeout_sec),
        "--model",
        model,
        "--disable-bot-cache",
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "command.json").write_text(json.dumps({"cmd": cmd, "env_flags": {
        DIRECT_PATH_PILOT_CONFIG_ENV: env[DIRECT_PATH_PILOT_CONFIG_ENV],
        ANCHORED_BARE_GRADE_ENV: env[ANCHORED_BARE_GRADE_ENV],
        MEMORY_PROVENANCE_ENV: env[MEMORY_PROVENANCE_ENV],
        QUESTION_INSTEAD_OF_HANDOFF_ENV: env[QUESTION_INSTEAD_OF_HANDOFF_ENV],
    }}, ensure_ascii=False, indent=2), encoding="utf-8")
    proc = subprocess.run(cmd, cwd=Path.cwd(), env=env, text=True, capture_output=True, check=False)
    (out_dir / "stdout.txt").write_text(proc.stdout or "", encoding="utf-8")
    (out_dir / "stderr.txt").write_text(proc.stderr or "", encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"dynamic replay failed rc={proc.returncode}; see {out_dir}")


def summarize_dynamic_run(run_dir: Path) -> dict[str, Any]:
    summary_path = run_dir / "dynamic_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    q = summary.get("question_instead_of_handoff") if isinstance(summary.get("question_instead_of_handoff"), Mapping) else {}
    llm_calls = summary.get("llm_calls") if isinstance(summary.get("llm_calls"), Mapping) else {}
    totals = summary.get("totals") if isinstance(summary.get("totals"), Mapping) else {}
    return {
        "ran": True,
        "run_dir": str(run_dir),
        "dialogs": totals.get("dialogs"),
        "turns": totals.get("turns"),
        "verdict_counts": summary.get("verdicts"),
        "hard_gate_failures": totals.get("hard_gate_failures"),
        "violated_gates": summary.get("violated_gates"),
        "question_instead_of_handoff": dict(q),
        "llm_calls": dict(llm_calls),
        "llm_calls_total": int(llm_calls.get("total") or 0),
        "config_validity": summary.get("config_validity"),
    }


def semantic_review(run_dir: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for dialog in read_jsonl(run_dir / "dynamic_dialog_transcripts.jsonl"):
        if not isinstance(dialog, Mapping):
            continue
        brand = str(dialog.get("brand") or "").strip().lower()
        for turn in dialog.get("turns") or []:
            if not isinstance(turn, Mapping):
                continue
            q = turn.get("bot_question_instead_of_handoff")
            text = str(turn.get("bot_text") or "")
            lower = text.casefold().replace("ё", "е")
            fired = isinstance(q, Mapping) and q.get("status") == "fired"
            rows.append(
                {
                    "dialog_id": dialog.get("dialog_id"),
                    "brand": brand,
                    "slot": q.get("slot") if isinstance(q, Mapping) else "",
                    "fired": fired,
                    "turns": len(dialog.get("turns") or []),
                    "mentions_other_brand": bool(
                        (brand == "unpk" and "фотон" in lower)
                        or (brand == "foton" and ("унпк" in lower or "мфти" in lower))
                    ),
                    "p0_text": bool(re.search(r"верните|жалоб|списал|претензи|возврат", lower, re.I)),
                    "bot_text": text,
                }
            )
    fired_rows = [row for row in rows if row["fired"]]
    issues = []
    if not fired_rows:
        issues.append("fired_zero")
    if any(row["turns"] > 1 for row in rows):
        issues.append("multi_turn_cycle_risk")
    if any(row["mentions_other_brand"] for row in rows):
        issues.append("brand_leak")
    if any(row["p0_text"] for row in fired_rows):
        issues.append("p0_question_risk")
    return {
        "verdict": "PASS_WITH_NOTES" if not issues else "BLOCKED",
        "issues": issues,
        "rows_total": len(rows),
        "fired": len(fired_rows),
        "examples": [
            {
                "dialog_id": row["dialog_id"],
                "brand": row["brand"],
                "slot": row["slot"],
                "bot_text": row["bot_text"],
            }
            for row in fired_rows[:5]
        ],
    }


def stable_dialog_id(source_path: Path, dialog: Mapping[str, Any], turn: Mapping[str, Any]) -> str:
    base = f"{source_path.parent.name}_{dialog.get('dialog_id')}_t{turn.get('turn') or 1}"
    value = re.sub(r"[^a-zA-Z0-9_]+", "_", base).strip("_").lower()
    return f"tz123_tz124_{value}"[:120]


def normalize_message(value: str) -> str:
    return " ".join(str(value or "").casefold().split())


def read_jsonl(path: Path) -> Iterable[Any]:
    if not path.exists():
        return
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            yield json.loads(line)


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
