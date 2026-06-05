#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shlex
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from mango_mvp.channels.subscription_llm import (
    SubscriptionDraftResult,
    SubscriptionLlmDraftProvider,
    build_codex_exec_command,
    codex_isolation_cwd,
    normalize_subscription_draft_payload,
    strip_internal_service_markers,
)
from mango_mvp.channels.telegram_pilot_context_builder import build_telegram_pilot_context_from_snapshot
from mango_mvp.channels.subscription_llm import AUTONOMY_MATRIX_SAFE_TOPIC_IDS
from mango_mvp.channels.new_lead_funnel import build_lead_funnel_state, lead_funnel_context_payload
from mango_mvp.channels.dialogue_memory import build_dialogue_memory, update_dialogue_memory_after_answer
from mango_mvp.channels.fact_retrieval import key_matches
from mango_mvp.channels.fact_claim_audit import FACT_AUDIT_VERSION as JUDGE_FACT_AUDIT_VERSION, audit_fact_claims as audit_fact_claims_for_judge
from mango_mvp.insights.tone_score import summarize_tone_scores


DEFAULT_V7_PATH = Path("/Users/dmitrijfabarisov/Claude Projects/Foton/mega_smoke_tests_v7_dynamic_sim_2026-05-21/v7_dynamic_client_sim_2026-05-21.jsonl")
DEFAULT_SNAPSHOT = Path("product_data/knowledge_base/kb_release_20260603_v6_5_summer_format_cleanup/kb_release_v3_snapshot.json")
DEFAULT_OUT_DIR = Path("audits/_inbox/telegram_dynamic_client_sim_v7")
SCHEMA_VERSION = "telegram_dynamic_client_sim_v1_2026_05_21"
METRIC_TARGETS = {
    "pass_rate": 0.8,
    "hard_gate_pass_rate": 0.95,
    "send_unedited_rate": 0.45,
    "avg_human_tone_score": 65.0,
}
HANDOFF_TRACE_ENV = "TELEGRAM_HANDOFF_TRACE"


@dataclass(frozen=True)
class DynamicSimInput:
    simulator_spec: Mapping[str, Any]
    judge_spec: Mapping[str, Any]
    personas: tuple[Mapping[str, Any], ...]


class FakeClientModel:
    def generate(self, prompt: str) -> Mapping[str, Any]:
        if "turn_index=1" in prompt:
            return {"message": "Здравствуйте, подойдёт ли ребёнку, если он отстаёт?", "stop": False}
        if "turn_index=2" in prompt:
            return {"message": "6 класс, математика, онлайн. Что дальше?", "stop": False}
        return {"message": "Спасибо, поняла.", "stop": True}


class FakeJudgeModel:
    def generate(self, prompt: str) -> Mapping[str, Any]:
        return {
            "dialog_id": "fake",
            "brand": "unpk",
            "hard_gates_passed": True,
            "violated_gates": [],
            "soft_flags_present": ["assumed_unstated_need"],
            "quality_scores": {
                "answered_the_question": 1,
                "no_reask_known": 1,
                "context_retention": 1,
                "human_tone": 1,
                "grammar_coherence": 2,
                "sales_progress": 1,
            },
            "human_tone_score_0_100": 70,
            "first_failing_turn": None,
            "lead_captured": False,
            "verdict": "PASS_WITH_NOTES",
            "rationale": "Fake judge result.",
        }


class FakeMemoryModel:
    def generate(self, prompt: str) -> Mapping[str, Any]:
        return {
            "slots": {"grade": "6", "subject": "математика", "format": "онлайн"},
            "topic": {"grade": "6", "subject": "математика", "format": "онлайн", "product_family": "regular_course"},
            "open_question": {"text": "Что дальше?", "kind": "other", "answered": False},
            "commitments": ["manager_handoff"],
            "summary": "Fake memory: клиент интересуется онлайн-математикой для 6 класса; бот передал вопрос менеджеру.",
        }


class FakeSemanticMatchModel:
    def generate(self, prompt: str) -> Mapping[str, Any]:
        return {"covers": True, "same_product": True, "reason": "fake semantic match"}


class FakeSellingComposeModel:
    def generate(self, prompt: str) -> Mapping[str, Any]:
        return {"text": "Понимаю, сумма важная. Опираюсь только на подтверждённые условия; менеджер поможет выбрать удобный шаг."}


class FakeBotProvider:
    def build_draft(self, client_message: str, *, context: Optional[Mapping[str, Any]] = None) -> SubscriptionDraftResult:
        return normalize_subscription_draft_payload(
            {
                "message_type": "question",
                "broad_group": "support",
                "topic_id": "service:S5_general_consultation",
                "confidence_theme": 0.8,
                "confidence_group": 0.8,
                "risk_level": "low",
                "route": "draft_for_manager",
                "draft_text": "Да, поможем подобрать программу. Напишите класс и предмет, если ещё не писали.",
                "safety_flags": ["manager_approval_required", "no_auto_send"],
            }
        )


class LlmCallCounter:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counts: Counter[str] = Counter()

    def increment(self, role: str) -> None:
        value = str(role or "").strip()
        if not value:
            return
        with self._lock:
            self._counts[value] += 1

    def snapshot(self) -> Mapping[str, int]:
        with self._lock:
            return dict(self._counts)


class CountingGenerateModel:
    def __init__(self, model: Any, *, role: str, counter: LlmCallCounter | None) -> None:
        self._model = model
        self._role = role
        self._counter = counter

    def generate(self, prompt: str) -> Mapping[str, Any]:
        if self._counter is not None:
            self._counter.increment(self._role)
        return self._model.generate(prompt)


def maybe_counting_model(model: Any, *, role: str, counter: LlmCallCounter | None) -> Any:
    if counter is None:
        return model
    return CountingGenerateModel(model, role=role, counter=counter)


class CountingSubscriptionLlmDraftProvider(SubscriptionLlmDraftProvider):
    def __init__(self, *args: Any, llm_call_counter: LlmCallCounter | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._llm_call_counter = llm_call_counter

    def _count_llm_call(self, role: str) -> None:
        if self._llm_call_counter is not None:
            self._llm_call_counter.increment(role)

    def _dialogue_contract_understanding_runner(self, prompt: str) -> Mapping[str, Any]:
        self._count_llm_call("bot_draft")
        return super()._dialogue_contract_understanding_runner(prompt)

    def _dialogue_contract_draft_runner(self, prompt: str) -> str:
        self._count_llm_call("bot_draft")
        return super()._dialogue_contract_draft_runner(prompt)

    def _dialogue_contract_repair_runner(self, prompt: str) -> str:
        self._count_llm_call("bot_draft")
        return super()._dialogue_contract_repair_runner(prompt)

    def _dialogue_contract_warmth_runner(self, prompt: str) -> str:
        self._count_llm_call("bot_draft")
        return super()._dialogue_contract_warmth_runner(prompt)

    def _dialogue_contract_faithfulness_runner(self, prompt: str) -> Mapping[str, Any] | str:
        self._count_llm_call("bot_critic")
        return super()._dialogue_contract_faithfulness_runner(prompt)

    def _dialogue_contract_semantic_match_runner(self, prompt: str) -> Mapping[str, Any] | str:
        self._count_llm_call("bot_critic")
        return super()._dialogue_contract_semantic_match_runner(prompt)

    def _semantic_diagnosis_guard_runner(self, prompt: str) -> Mapping[str, Any] | str:
        self._count_llm_call("bot_diagnosis_guard")
        return super()._semantic_diagnosis_guard_runner(prompt)

    def _answer_quality_llm_rewrite_runner(
        self,
        *,
        result: SubscriptionDraftResult,
        client_message: str,
        context: Mapping[str, Any] | None,
        assessment: Any,
    ) -> Mapping[str, Any]:
        self._count_llm_call("bot_draft")
        return super()._answer_quality_llm_rewrite_runner(
            result=result,
            client_message=client_message,
            context=context,
            assessment=assessment,
        )

    def _humanity_x2_rewrite_runner(self, prompt: str) -> str:
        self._count_llm_call("bot_draft")
        return super()._humanity_x2_rewrite_runner(prompt)

    def _run_once(self, prompt: str, *, force_manager_only: bool) -> SubscriptionDraftResult:
        self._count_llm_call("bot_draft")
        return super()._run_once(prompt, force_manager_only=force_manager_only)


class CodexJsonModel:
    def __init__(
        self,
        *,
        model: str,
        reasoning_effort: str,
        timeout_sec: int,
        codex_bin: str = "codex",
        isolated: bool = False,
    ) -> None:
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.timeout_sec = timeout_sec
        self.codex_bin = codex_bin
        self.isolated = bool(isolated)

    def generate(self, prompt: str) -> Mapping[str, Any]:
        with tempfile.NamedTemporaryFile(prefix="mango_dynamic_sim_", suffix=".json") as out_file:
            output_path = Path(out_file.name)
            with codex_isolation_cwd(self.isolated) as isolated_cwd:
                cmd = build_codex_exec_command(
                    output_path=output_path,
                    codex_bin=self.codex_bin,
                    model=self.model,
                    reasoning_effort=self.reasoning_effort,
                    isolated=self.isolated,
                    cwd=isolated_cwd,
                )
                proc = subprocess.run(
                    cmd,
                    input=prompt,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_sec,
                    check=False,
                    env=_codex_env(),
                )
            raw = output_path.read_text(encoding="utf-8", errors="ignore")
        if proc.returncode != 0:
            raise RuntimeError(f"codex exec failed rc={proc.returncode}: {(proc.stderr or '')[-500:]}")
        return extract_json_object(raw or proc.stdout or proc.stderr)


_CLAUDE_DEFAULT_MODEL = "claude-sonnet-4-6"
_CLAUDE_REASONING_ARGS: Mapping[str, tuple[str, ...]] = {
    "low": ("--effort", "low"),
    "medium": ("--effort", "medium"),
    "high": ("--effort", "high"),
    "xhigh": ("--effort", "xhigh"),
    "max": ("--effort", "max"),
}
_CLAUDE_AUTH_MODES = frozenset({"subscription", "bare"})
_EMPTY_CLAUDE_MCP_CONFIG = '{"mcpServers":{}}'


def _claude_reasoning_args(level: str) -> list[str]:
    value = str(level or "").strip().lower()
    if value in {"", "none", "off"}:
        return []
    return list(_CLAUDE_REASONING_ARGS.get(value, _CLAUDE_REASONING_ARGS["high"]))


def _claude_env(base_env: Mapping[str, str] | None = None) -> dict[str, str]:
    return dict(os.environ if base_env is None else base_env)


def build_claude_print_command(
    *,
    claude_bin: str = "claude",
    model: str = _CLAUDE_DEFAULT_MODEL,
    reasoning_effort: str = "high",
    auth_mode: str = "subscription",
) -> list[str]:
    normalized_auth_mode = str(auth_mode or "subscription").strip().lower()
    if normalized_auth_mode not in _CLAUDE_AUTH_MODES:
        raise ValueError(f"Unsupported claude auth mode: {auth_mode!r}")
    cmd = [
        str(claude_bin or "claude").strip() or "claude",
        "-p",
        "--model",
        str(model or _CLAUDE_DEFAULT_MODEL).strip() or _CLAUDE_DEFAULT_MODEL,
        "--output-format",
        "text",
        "--tools",
        "",
        "--strict-mcp-config",
        "--mcp-config",
        _EMPTY_CLAUDE_MCP_CONFIG,
        "--no-session-persistence",
        "--disable-slash-commands",
        "--permission-mode",
        "plan",
    ]
    if normalized_auth_mode == "bare":
        cmd.insert(2, "--bare")
    cmd.extend(_claude_reasoning_args(reasoning_effort))
    return cmd


class ClaudeJsonModel:
    def __init__(
        self,
        *,
        model: str = _CLAUDE_DEFAULT_MODEL,
        reasoning_effort: str = "high",
        timeout_sec: int = 180,
        claude_bin: str = "claude",
        auth_mode: str = "subscription",
    ) -> None:
        self.model = str(model or _CLAUDE_DEFAULT_MODEL).strip() or _CLAUDE_DEFAULT_MODEL
        self.reasoning_effort = reasoning_effort
        self.timeout_sec = timeout_sec
        self.claude_bin = str(claude_bin or "claude").strip() or "claude"
        self.auth_mode = str(auth_mode or "subscription").strip().lower() or "subscription"

    def generate(self, prompt: str) -> Mapping[str, Any]:
        cmd = build_claude_print_command(
            claude_bin=self.claude_bin,
            model=self.model,
            reasoning_effort=self.reasoning_effort,
            auth_mode=self.auth_mode,
        )
        proc = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=self.timeout_sec,
            check=False,
            env=_claude_env(),
        )
        if proc.returncode != 0:
            raise RuntimeError(f"claude -p failed rc={proc.returncode}: {(proc.stderr or '')[-500:]}")
        return extract_json_object(proc.stdout or proc.stderr)


class ClaudeCliRunner:
    def __init__(
        self,
        *,
        model: str = _CLAUDE_DEFAULT_MODEL,
        reasoning_effort: str = "high",
        timeout_sec: int = 180,
        claude_bin: str = "claude",
        auth_mode: str = "subscription",
    ) -> None:
        self.model = str(model or _CLAUDE_DEFAULT_MODEL).strip() or _CLAUDE_DEFAULT_MODEL
        self.reasoning_effort = reasoning_effort
        self.timeout_sec = timeout_sec
        self.claude_bin = str(claude_bin or "claude").strip() or "claude"
        self.auth_mode = str(auth_mode or "subscription").strip().lower() or "subscription"
        self.last_commands: list[list[str]] = []
        self._events: list[dict[str, Any]] = []
        self._events_lock = threading.Lock()

    def drain_events(self) -> list[dict[str, Any]]:
        with self._events_lock:
            events = list(self._events)
            self._events.clear()
        return events

    def __call__(
        self,
        _cmd: Sequence[str],
        *,
        input: str,
        capture_output: bool,
        text: bool,
        check: bool,
        timeout: int,
        env: Mapping[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        cmd = build_claude_print_command(
            claude_bin=self.claude_bin,
            model=self.model,
            reasoning_effort=self.reasoning_effort,
            auth_mode=self.auth_mode,
        )
        self.last_commands.append(cmd)
        proc = subprocess.run(
            cmd,
            input=input,
            capture_output=capture_output,
            text=text,
            check=check,
            timeout=timeout or self.timeout_sec,
            env=_claude_env(env),
        )
        event = _claude_cli_event_if_visible_failure(
            requested_cmd=_cmd,
            actual_cmd=cmd,
            returncode=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
            prompt=input,
        )
        if event:
            with self._events_lock:
                self._events.append(event)
            print(_format_claude_cli_event_log(event), file=sys.stderr, flush=True)
        return subprocess.CompletedProcess(cmd, proc.returncode, stdout=proc.stdout, stderr=proc.stderr)


def _claude_cli_event_if_visible_failure(
    *,
    requested_cmd: Sequence[str],
    actual_cmd: Sequence[str],
    returncode: int,
    stdout: str | None,
    stderr: str | None,
    prompt: str,
) -> dict[str, Any]:
    stdout_text = str(stdout or "")
    stderr_text = str(stderr or "")
    if returncode == 0 and (stdout_text.strip() or stderr_text.strip()):
        return {}
    reason = "nonzero_returncode" if returncode != 0 else "empty_output"
    return {
        "reason": reason,
        "stage": _claude_cli_stage_from_requested_cmd(requested_cmd),
        "returncode": int(returncode),
        "cmd": _shell_join(actual_cmd),
        "stdout_tail": _tail_compact(stdout_text),
        "stderr_tail": _tail_compact(stderr_text),
        "prompt_chars": len(str(prompt or "")),
    }


def _claude_cli_stage_from_requested_cmd(cmd: Sequence[str]) -> str:
    values = [str(item) for item in cmd]
    try:
        output_path = values[values.index("--output-last-message") + 1]
    except (ValueError, IndexError):
        return ""
    name = Path(output_path).name
    name = re.sub(r"^[A-Za-z0-9]+_", "", name)
    name = re.sub(r"_[A-Za-z0-9]+(?:\.[^.]+)?$", "", name)
    return name


def _shell_join(cmd: Sequence[str]) -> str:
    return shlex.join(str(part) for part in cmd)


def _tail_compact(text: str, *, limit: int = 1200) -> str:
    compact = " ".join(str(text or "").split())
    return compact[-limit:]


def _format_claude_cli_event_log(event: Mapping[str, Any]) -> str:
    return (
        "claude_cli_error="
        + json.dumps(
            {
                "stage": event.get("stage") or "",
                "reason": event.get("reason") or "",
                "returncode": event.get("returncode"),
                "stderr_tail": event.get("stderr_tail") or "",
                "stdout_tail": event.get("stdout_tail") or "",
                "cmd": event.get("cmd") or "",
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


def _consume_claude_cli_events(bot_provider: Any) -> list[dict[str, Any]]:
    runner = getattr(bot_provider, "_dynamic_sim_claude_runner", None)
    if runner is None or not hasattr(runner, "drain_events"):
        return []
    try:
        return list(runner.drain_events())
    except Exception:  # noqa: BLE001
        return []


def _claude_model_from_args(args: argparse.Namespace) -> str:
    explicit = str(getattr(args, "claude_model", "") or "").strip()
    if explicit:
        return explicit
    model = str(getattr(args, "model", "") or "").strip()
    if model.startswith("claude"):
        return model
    return _CLAUDE_DEFAULT_MODEL


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run dynamic client simulator against Telegram bot draft logic.")
    parser.add_argument("--scenarios", type=Path, default=DEFAULT_V7_PATH)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--brand", choices=("all", "foton", "unpk"), default="all")
    parser.add_argument("--limit", type=int, default=0, help="Limit personas for smoke runs.")
    parser.add_argument("--max-turns", type=int, default=0, help="Override persona max_turns.")
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Run this many dialogs concurrently. Recommended: 2-3 for codex modes, higher only for fake smoke.",
    )
    parser.add_argument(
        "--transcripts-in",
        type=Path,
        default=None,
        help="Re-judge existing dynamic_dialog_transcripts.jsonl instead of re-running client and bot.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue an interrupted run by loading existing dynamic_dialog_transcripts.jsonl from --out-dir.",
    )
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        help="When resuming, keep completed dialogs and run only missing/incomplete dialogs.",
    )
    parser.add_argument(
        "--only-failed",
        action="store_true",
        help="When resuming, rerun only dialogs with FAIL verdict or previous infrastructure errors.",
    )
    parser.add_argument(
        "--only-timeout",
        action="store_true",
        help="When resuming, rerun only dialogs that ended with timeout.",
    )
    parser.add_argument("--client-mode", choices=("codex", "fake"), default="codex")
    parser.add_argument("--judge-mode", choices=("codex", "fake"), default="codex")
    parser.add_argument("--bot-mode", choices=("codex", "claude", "fake"), default="codex")
    parser.add_argument(
        "--codex-isolated",
        dest="codex_isolated",
        action="store_true",
        default=True,
        help="Run Codex bot-side calls without user config/rules in a clean temporary cwd. Default for honest tone A/B.",
    )
    parser.add_argument(
        "--no-codex-isolated",
        dest="codex_isolated",
        action="store_false",
        help="Use current Codex user config for GPT bot calls; intended only as baseline A.",
    )
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--claude-model", default=_CLAUDE_DEFAULT_MODEL)
    parser.add_argument("--claude-bin", default="claude")
    parser.add_argument("--claude-auth-mode", choices=tuple(sorted(_CLAUDE_AUTH_MODES)), default="subscription")
    parser.add_argument("--bot-reasoning", default="medium")
    parser.add_argument("--client-reasoning", default="medium")
    parser.add_argument("--judge-reasoning", default="high")
    parser.add_argument("--memory-mode", choices=("codex", "fake", "off"), default="codex")
    parser.add_argument("--memory-model", default="gpt-5.5")
    parser.add_argument("--memory-reasoning", default="low")
    parser.add_argument("--semantic-mode", choices=("codex", "fake", "off"), default="codex")
    parser.add_argument("--semantic-model", default="gpt-5.5")
    parser.add_argument("--semantic-reasoning", default="medium")
    parser.add_argument("--selling-mode", choices=("gen", "det"), default=os.getenv("TELEGRAM_A_SELLING_MODE", "gen"))
    parser.add_argument("--selling-model", default="gpt-5.5")
    parser.add_argument("--selling-reasoning", default="medium")
    parser.add_argument("--selling-compose-fake", action="store_true", help="Use deterministic fake selling composer for smoke tests.")
    parser.add_argument("--timeout-sec", type=int, default=180)
    parser.add_argument(
        "--disable-bot-cache",
        action="store_true",
        help="Do not use cached bot LLM drafts; useful when validating fresh prompt/code changes.",
    )
    parser.add_argument(
        "--enable-llm-rewriter",
        action="store_true",
        help="Enable optional answer-quality LLM rewrite layer for bot answers in this run only.",
    )
    args = parser.parse_args(argv)
    llm_call_counter = LlmCallCounter()
    setattr(args, "llm_call_counter", llm_call_counter)
    if args.parallel < 1:
        raise ValueError("--parallel must be >= 1")
    if args.enable_llm_rewriter:
        os.environ["TELEGRAM_ANSWER_QUALITY_LLM_REWRITE"] = "1"
    if args.memory_mode == "codex" and str(args.memory_reasoning or "").strip().lower() != "low":
        raise ValueError("--memory-reasoning must be low for codex memory mode")
    os.environ["TELEGRAM_A_SELLING_MODE"] = str(args.selling_mode or "gen")

    if "stable_runtime" in args.out_dir.resolve(strict=False).parts:
        raise ValueError("Refusing to write dynamic sim outputs under stable_runtime")
    if (args.bot_mode == "codex" or args.transcripts_in is not None) and not args.snapshot.exists():
        raise FileNotFoundError(f"Knowledge snapshot not found: {args.snapshot}")

    sim_input = load_dynamic_sim_input(args.scenarios)
    personas = [item for item in sim_input.personas if args.brand == "all" or item.get("brand") == args.brand]
    if args.limit > 0:
        personas = personas[: args.limit]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    transcripts_path = args.out_dir / "dynamic_dialog_transcripts.jsonl"
    judge_path = args.out_dir / "dynamic_judge_results.jsonl"
    turns_path = args.out_dir / "dynamic_turns.csv"
    review_queue_path = args.out_dir / "human_review_queue.csv"
    full_transcripts_md_path = args.out_dir / "full_transcripts.md"
    per_dialog_dir = args.out_dir / "transcripts_md"
    summary_path = args.out_dir / "dynamic_summary.json"
    summary_md_path = args.out_dir / "dynamic_summary.md"
    persona_order = {
        str(persona.get("dialog_id") or ""): index
        for index, persona in enumerate(personas)
    }

    if args.transcripts_in is not None:
        judge_model = build_judge_model(args)
        memory_model = build_memory_model(args)
        transcripts = [
            attach_context_facts_to_dialog(dialog, snapshot_path=args.snapshot, memory_model=memory_model)
            for dialog in load_transcripts(args.transcripts_in)
            if args.brand == "all" or dialog.get("brand") == args.brand
        ]
        if args.limit > 0:
            transcripts = transcripts[: args.limit]
        judge_results = []
        for dialog in transcripts:
            judge = judge_model.generate(build_judge_prompt(sim_input.judge_spec, dialog.get("persona") or {}, dialog.get("turns") or []))
            judge_results.append(normalize_judge_result(judge, dialog_id=str(dialog.get("dialog_id") or ""), brand=str(dialog.get("brand") or "")))
        transcripts = [{**dialog, "judge_result": judge} for dialog, judge in zip(transcripts, judge_results)]
        turn_rows = build_turn_rows(transcripts)
    else:
        transcripts = []
        if args.resume and transcripts_path.exists():
            transcripts = list(load_transcripts(transcripts_path))
            print(f"resume_loaded_dialogs={len(transcripts)}", flush=True)
        existing_by_id = {str(dialog.get("dialog_id") or ""): dict(dialog) for dialog in transcripts if str(dialog.get("dialog_id") or "").strip()}
        rerun_ids = {
            dialog_id
            for dialog_id, dialog in existing_by_id.items()
            if _should_rerun_existing_dialog(dialog, only_failed=args.only_failed, only_timeout=args.only_timeout)
        }
        if rerun_ids:
            transcripts = [dialog for dialog in transcripts if str(dialog.get("dialog_id") or "") not in rerun_ids]
            existing_by_id = {str(dialog.get("dialog_id") or ""): dict(dialog) for dialog in transcripts if str(dialog.get("dialog_id") or "").strip()}
        completed_ids = {
            dialog_id
            for dialog_id, dialog in existing_by_id.items()
            if _dialog_completed(dialog) and (args.resume or args.skip_completed)
        }
        judge_results = [
            dict(dialog.get("judge_result") or {})
            for dialog in transcripts
            if isinstance(dialog.get("judge_result"), Mapping)
        ]
        pending_personas = []
        for persona in personas:
            dialog_id = str(persona.get("dialog_id") or "")
            if dialog_id in completed_ids:
                print(f"skip_completed_dialog={dialog_id}", flush=True)
                continue
            if (args.only_failed or args.only_timeout) and dialog_id not in rerun_ids:
                continue
            pending_personas.append(persona)

        if args.parallel == 1:
            client_model = build_client_model(args)
            judge_model = build_judge_model(args)
            bot_provider = build_bot_provider(args)
            memory_model = build_memory_model(args)
            selling_compose_model = build_selling_compose_model(args)
            for persona in pending_personas:
                dialog_id = str(persona.get("dialog_id") or "")
                print(f"run_dialog={dialog_id}", flush=True)
                started = time.time()
                try:
                    dialog = run_one_dialog(
                        persona,
                        simulator_spec=sim_input.simulator_spec,
                        judge_spec=sim_input.judge_spec,
                        client_model=client_model,
                        judge_model=judge_model,
                        bot_provider=bot_provider,
                        memory_model=memory_model,
                        selling_compose_model=selling_compose_model,
                        snapshot_path=args.snapshot,
                        max_turns_override=args.max_turns,
                        debug_trace_run_dir=args.out_dir,
                    )
                    dialog = {**dialog, "elapsed_seconds": round(time.time() - started, 3), "run_status": "completed"}
                except Exception as exc:  # noqa: BLE001
                    dialog = build_infra_error_dialog(
                        persona,
                        exc,
                        elapsed_seconds=round(time.time() - started, 3),
                    )
                transcripts.append(dialog)
                transcripts = sort_transcripts_by_persona_order(transcripts, persona_order)
                judge_results = extract_judge_results(transcripts)
                completed_ids.add(str(dialog.get("dialog_id") or ""))
                write_dynamic_outputs(
                    transcripts,
                    judge_results,
                    transcripts_path=transcripts_path,
                    judge_path=judge_path,
                    turns_path=turns_path,
                    review_queue_path=review_queue_path,
                    full_transcripts_md_path=full_transcripts_md_path,
                    per_dialog_dir=per_dialog_dir,
                    summary_path=summary_path,
                    summary_md_path=summary_md_path,
                    scenario_path=args.scenarios,
                    snapshot_path=args.snapshot,
                    judge_spec=sim_input.judge_spec,
                    parallel=args.parallel,
                    llm_calls=llm_call_counter.snapshot(),
                )
                print(
                    f"done_dialog={dialog_id} elapsed={dialog['elapsed_seconds']}s verdict={dialog['judge_result'].get('verdict')}",
                    flush=True,
                )
        else:
            print(f"parallel_dialogs={args.parallel} pending={len(pending_personas)}", flush=True)
            with ThreadPoolExecutor(max_workers=args.parallel) as executor:
                future_to_persona = {
                    executor.submit(
                        run_one_dialog_isolated,
                        persona,
                        simulator_spec=sim_input.simulator_spec,
                        judge_spec=sim_input.judge_spec,
                        snapshot_path=args.snapshot,
                        max_turns_override=args.max_turns,
                        args=args,
                    ): persona
                    for persona in pending_personas
                }
                for future in as_completed(future_to_persona):
                    persona = future_to_persona[future]
                    dialog_id = str(persona.get("dialog_id") or "")
                    try:
                        dialog = future.result()
                    except Exception as exc:  # noqa: BLE001
                        dialog = build_infra_error_dialog(persona, exc, elapsed_seconds=None)
                    transcripts.append(dialog)
                    transcripts = sort_transcripts_by_persona_order(transcripts, persona_order)
                    judge_results = extract_judge_results(transcripts)
                    completed_ids.add(str(dialog.get("dialog_id") or ""))
                    write_dynamic_outputs(
                        transcripts,
                        judge_results,
                        transcripts_path=transcripts_path,
                        judge_path=judge_path,
                        turns_path=turns_path,
                        review_queue_path=review_queue_path,
                        full_transcripts_md_path=full_transcripts_md_path,
                        per_dialog_dir=per_dialog_dir,
                        summary_path=summary_path,
                        summary_md_path=summary_md_path,
                        scenario_path=args.scenarios,
                        snapshot_path=args.snapshot,
                        judge_spec=sim_input.judge_spec,
                        parallel=args.parallel,
                        llm_calls=llm_call_counter.snapshot(),
                    )
                    print(
                        f"done_dialog={dialog_id} elapsed={dialog.get('elapsed_seconds')}s verdict={dialog['judge_result'].get('verdict')}",
                        flush=True,
                    )
        turn_rows = build_turn_rows(transcripts)

    summary = write_dynamic_outputs(
        transcripts,
        judge_results,
        transcripts_path=transcripts_path,
        judge_path=judge_path,
        turns_path=turns_path,
        review_queue_path=review_queue_path,
        full_transcripts_md_path=full_transcripts_md_path,
        per_dialog_dir=per_dialog_dir,
        summary_path=summary_path,
        summary_md_path=summary_md_path,
        scenario_path=args.scenarios,
        snapshot_path=args.snapshot,
        judge_spec=sim_input.judge_spec,
        parallel=args.parallel,
        llm_calls=llm_call_counter.snapshot(),
    )
    print(json.dumps({"ok": True, "out_dir": str(args.out_dir), **summary["totals"]}, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def write_dynamic_outputs(
    transcripts: Sequence[Mapping[str, Any]],
    judge_results: Sequence[Mapping[str, Any]],
    *,
    transcripts_path: Path,
    judge_path: Path,
    turns_path: Path,
    review_queue_path: Path,
    full_transcripts_md_path: Path,
    per_dialog_dir: Path,
    summary_path: Path,
    summary_md_path: Path,
    scenario_path: Path,
    snapshot_path: Path,
    judge_spec: Mapping[str, Any] | None = None,
    parallel: int = 1,
    llm_calls: Mapping[str, int] | None = None,
) -> Mapping[str, Any]:
    write_jsonl(transcripts_path, transcripts)
    write_jsonl(judge_path, judge_results)
    write_csv(turns_path, build_turn_rows(transcripts))
    write_human_review_csv(review_queue_path, build_human_review_rows(transcripts, judge_results))
    full_transcripts_md_path.write_text(render_full_transcripts_md(transcripts), encoding="utf-8")
    per_dialog_dir.mkdir(parents=True, exist_ok=True)
    for dialog in transcripts:
        dialog_path = per_dialog_dir / f"{safe_filename(str(dialog.get('dialog_id') or 'dialog'))}.md"
        dialog_path.write_text(render_one_dialog_md(dialog), encoding="utf-8")
    summary = build_summary(
        transcripts,
        judge_results,
        scenario_path=scenario_path,
        snapshot_path=snapshot_path,
        judge_spec=judge_spec,
        parallel=parallel,
        llm_calls=llm_calls,
    )
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    summary_md_path.write_text(render_summary_md(summary), encoding="utf-8")
    return summary


def load_dynamic_sim_input(path: Path) -> DynamicSimInput:
    simulator_spec: Mapping[str, Any] | None = None
    judge_spec: Mapping[str, Any] | None = None
    personas: list[Mapping[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        item = json.loads(line)
        if item.get("type") == "simulator_spec":
            simulator_spec = item
        elif item.get("type") == "judge_spec":
            judge_spec = item
        elif item.get("type") == "persona":
            personas.append(item)
        else:
            raise ValueError(f"Unknown row type at line {line_no}: {item.get('type')}")
    if simulator_spec is None or judge_spec is None or not personas:
        raise ValueError("Scenarios must include simulator_spec, judge_spec, and persona rows")
    return DynamicSimInput(simulator_spec=simulator_spec, judge_spec=judge_spec, personas=tuple(personas))


def load_transcripts(path: Path) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            payload = json.loads(line)
            if isinstance(payload, Mapping):
                rows.append(payload)
    return rows


def build_client_model(args: argparse.Namespace) -> Any:
    if args.client_mode == "fake":
        return FakeClientModel()
    return maybe_counting_model(
        CodexJsonModel(
            model=args.model,
            reasoning_effort=args.client_reasoning,
            timeout_sec=args.timeout_sec,
        ),
        role="client",
        counter=getattr(args, "llm_call_counter", None),
    )


def build_judge_model(args: argparse.Namespace) -> Any:
    if args.judge_mode == "fake":
        return FakeJudgeModel()
    return maybe_counting_model(
        CodexJsonModel(
            model=args.model,
            reasoning_effort=args.judge_reasoning,
            timeout_sec=args.timeout_sec,
        ),
        role="judge",
        counter=getattr(args, "llm_call_counter", None),
    )


def build_memory_model(args: argparse.Namespace) -> Any:
    if args.memory_mode == "off":
        return None
    if args.memory_mode == "fake":
        return FakeMemoryModel()
    return maybe_counting_model(
        CodexJsonModel(
            model=args.memory_model,
            reasoning_effort=args.memory_reasoning,
            timeout_sec=args.timeout_sec,
            codex_bin=getattr(args, "codex_bin", "codex"),
        ),
        role="memory",
        counter=getattr(args, "llm_call_counter", None),
    )


def build_semantic_match_model(args: argparse.Namespace) -> Any:
    if args.semantic_mode == "off":
        return None
    if args.semantic_mode == "fake":
        return FakeSemanticMatchModel()
    return maybe_counting_model(
        CodexJsonModel(
            model=args.semantic_model,
            reasoning_effort=args.semantic_reasoning,
            timeout_sec=args.timeout_sec,
            codex_bin=getattr(args, "codex_bin", "codex"),
        ),
        role="bot_critic",
        counter=getattr(args, "llm_call_counter", None),
    )


def build_selling_compose_model(args: argparse.Namespace) -> Any:
    if str(getattr(args, "selling_mode", "gen") or "gen") == "det":
        return None
    if getattr(args, "selling_compose_fake", False):
        return maybe_counting_model(
            FakeSellingComposeModel(),
            role="bot_selling_compose",
            counter=getattr(args, "llm_call_counter", None),
        )
    return maybe_counting_model(
        CodexJsonModel(
            model=args.selling_model,
            reasoning_effort=args.selling_reasoning,
            timeout_sec=args.timeout_sec,
            codex_bin=getattr(args, "codex_bin", "codex"),
            isolated=bool(getattr(args, "codex_isolated", False)),
        ),
        role="bot_selling_compose",
        counter=getattr(args, "llm_call_counter", None),
    )


def build_bot_provider(args: argparse.Namespace, *, dialog_id: str = "") -> Any:
    if args.bot_mode == "fake":
        return FakeBotProvider()
    if getattr(args, "disable_bot_cache", False):
        cache_dir = None
    else:
        cache_dir = Path(".codex_local/telegram_dynamic_client_sim/llm_cache")
        if dialog_id:
            cache_dir = cache_dir / safe_filename(dialog_id)
    semantic_match_model = build_semantic_match_model(args)
    runner = None
    model = args.model
    if args.bot_mode == "claude":
        model = _claude_model_from_args(args)
        runner = ClaudeCliRunner(
            model=model,
            reasoning_effort=args.bot_reasoning,
            timeout_sec=args.timeout_sec,
            claude_bin=getattr(args, "claude_bin", "claude"),
            auth_mode=getattr(args, "claude_auth_mode", "subscription"),
        )
    provider = CountingSubscriptionLlmDraftProvider(
        model=model,
        reasoning_effort=args.bot_reasoning,
        timeout_sec=args.timeout_sec,
        cache_dir=cache_dir,
        runner=runner,
        dialogue_contract_semantic_match_fn=semantic_match_model.generate if semantic_match_model is not None else None,
        dialogue_contract_semantic_match_enabled=semantic_match_model is not None,
        llm_call_counter=getattr(args, "llm_call_counter", None),
        codex_isolated=bool(getattr(args, "codex_isolated", True)) if args.bot_mode == "codex" else False,
    )
    if runner is not None:
        setattr(provider, "_dynamic_sim_claude_runner", runner)
    return provider


def run_one_dialog_isolated(
    persona: Mapping[str, Any],
    *,
    simulator_spec: Mapping[str, Any],
    judge_spec: Mapping[str, Any],
    snapshot_path: Path,
    max_turns_override: int,
    args: argparse.Namespace,
) -> Mapping[str, Any]:
    dialog_id = str(persona.get("dialog_id") or "")
    started = time.time()
    print(f"run_dialog={dialog_id}", flush=True)
    dialog = run_one_dialog(
        persona,
        simulator_spec=simulator_spec,
        judge_spec=judge_spec,
        client_model=build_client_model(args),
        judge_model=build_judge_model(args),
        bot_provider=build_bot_provider(args, dialog_id=dialog_id),
        memory_model=build_memory_model(args),
        selling_compose_model=build_selling_compose_model(args),
        snapshot_path=snapshot_path,
        max_turns_override=max_turns_override,
        debug_trace_run_dir=args.out_dir,
    )
    return {**dialog, "elapsed_seconds": round(time.time() - started, 3), "run_status": "completed"}


def _dialog_completed(dialog: Mapping[str, Any]) -> bool:
    status = str(dialog.get("run_status") or "").strip()
    if status:
        return status == "completed"
    return isinstance(dialog.get("judge_result"), Mapping) and bool(dialog.get("turns"))


def _should_rerun_existing_dialog(
    dialog: Mapping[str, Any],
    *,
    only_failed: bool,
    only_timeout: bool,
) -> bool:
    if only_timeout:
        return str(dialog.get("run_status") or "") == "timeout"
    if not only_failed:
        return False
    status = str(dialog.get("run_status") or "")
    judge = dialog.get("judge_result") if isinstance(dialog.get("judge_result"), Mapping) else {}
    return status in {"infra_error", "timeout"} or str(judge.get("verdict") or "") == "FAIL"


def build_infra_error_dialog(
    persona: Mapping[str, Any],
    exc: BaseException,
    *,
    elapsed_seconds: float | None,
) -> Mapping[str, Any]:
    dialog_id = str(persona.get("dialog_id") or "")
    brand = str(persona.get("brand") or "")
    status = "timeout" if isinstance(exc, (TimeoutError, subprocess.TimeoutExpired)) or "timeout" in str(exc).casefold() else "infra_error"
    error_text = f"{exc.__class__.__name__}: {str(exc)}"
    return {
        "dialog_id": dialog_id,
        "brand": brand,
        "persona": dict(persona),
        "turns": [],
        "run_status": status,
        "infra_error": error_text[-2000:],
        "elapsed_seconds": elapsed_seconds,
        "judge_result": {
            "dialog_id": dialog_id,
            "brand": brand,
            "verdict": "FAIL",
            "hard_gates_passed": False,
            "violated_gates": [status],
            "soft_flags_present": [],
            "human_tone_score_0_100": 0,
            "rationale": f"Диалог не был завершён из-за инфраструктурной ошибки: {error_text[-500:]}",
        },
    }


def sort_transcripts_by_persona_order(
    transcripts: Sequence[Mapping[str, Any]],
    persona_order: Mapping[str, int],
) -> list[Mapping[str, Any]]:
    return sorted(
        transcripts,
        key=lambda dialog: (
            persona_order.get(str(dialog.get("dialog_id") or ""), 10**9),
            str(dialog.get("dialog_id") or ""),
        ),
    )


def extract_judge_results(transcripts: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [
        dict(dialog.get("judge_result") or {})
        for dialog in transcripts
        if isinstance(dialog.get("judge_result"), Mapping)
    ]


def build_turn_rows(transcripts: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    for dialog in transcripts:
        for turn in dialog.get("turns") or []:
            rows.append(
                {
                    "dialog_id": dialog.get("dialog_id") or "",
                    "brand": dialog.get("brand") or "",
                    "turn": turn.get("turn"),
                    "client_message": turn.get("client_message") or "",
                    "bot_text": turn.get("bot_text") or "",
                    "bot_route": turn.get("bot_route") or "",
                    "bot_topic_id": turn.get("bot_topic_id") or "",
                    "bot_conversation_intent": (turn.get("bot_conversation_intent_plan") or {}).get("primary_intent")
                    if isinstance(turn.get("bot_conversation_intent_plan"), Mapping)
                    else "",
                    "bot_conversation_topic_switch": (turn.get("bot_conversation_intent_plan") or {}).get("topic_switch_decision")
                    if isinstance(turn.get("bot_conversation_intent_plan"), Mapping)
                    else "",
                    "bot_answer_contract_intent": (turn.get("bot_answer_contract") or {}).get("primary_intent")
                    if isinstance(turn.get("bot_answer_contract"), Mapping)
                    else "",
                    "bot_answer_contract_direct_question": (turn.get("bot_answer_contract") or {}).get("direct_question")
                    if isinstance(turn.get("bot_answer_contract"), Mapping)
                    else "",
                    "bot_answer_contract_p0_required": (turn.get("bot_answer_contract") or {}).get("p0_required")
                    if isinstance(turn.get("bot_answer_contract"), Mapping)
                    else "",
                    "bot_safety_flags": "|".join(str(flag) for flag in (turn.get("bot_safety_flags") or [])),
                    "bot_fallback_reason": turn.get("bot_fallback_reason") or "",
                    "bot_provider_error": turn.get("bot_provider_error") or "",
                    "bot_is_manager_deferral": bool(turn.get("bot_is_manager_deferral")),
                    "bot_reason_class": turn.get("bot_reason_class") or "",
                    "bot_reason_evidence": json.dumps(
                        turn.get("bot_reason_evidence") or {},
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    "bot_authoritative_output_gate": json.dumps(
                        turn.get("bot_authoritative_output_gate") or {},
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    "bot_claude_cli_error_count": int(turn.get("bot_claude_cli_error_count") or 0),
                    "bot_claude_cli_errors": json.dumps(
                        turn.get("bot_claude_cli_errors") or [],
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    "handoff_trace": json.dumps(turn.get("handoff_trace") or {}, ensure_ascii=False, sort_keys=True),
                    "bot_answer_quality_findings": "|".join(str(flag) for flag in (turn.get("bot_answer_quality_findings") or [])),
                    "bot_answer_quality_rewritten": turn.get("bot_answer_quality_rewritten"),
                    "judge_fact_audit_levels": "|".join(
                        str(item.get("level") or "")
                        for item in ((turn.get("judge_fact_audit") or {}).get("items") or [])
                        if isinstance(item, Mapping) and str(item.get("level") or "").strip()
                    )
                    if isinstance(turn.get("judge_fact_audit"), Mapping)
                    else "",
                    "context_parity_checked": turn.get("context_parity_checked"),
                    "client_stop": turn.get("client_stop"),
                }
            )
    return rows


def attach_context_facts_to_dialog(
    dialog: Mapping[str, Any],
    *,
    snapshot_path: Path,
    memory_model: Any = None,
) -> Mapping[str, Any]:
    persona = dialog.get("persona") if isinstance(dialog.get("persona"), Mapping) else {}
    recent_messages: list[str] = []
    turns: list[Mapping[str, Any]] = []
    dialogue_memory: Mapping[str, Any] = {}
    for raw_turn in dialog.get("turns") or []:
        turn = dict(raw_turn)
        client_message = str(turn.get("client_message") or "")
        bot_text = str(turn.get("bot_text") or "")
        context = build_bot_prompt_context(
            client_message,
            persona=persona,
            recent_messages=recent_messages,
            snapshot_path=snapshot_path,
            dialogue_memory=dialogue_memory,
        )
        turn["bot_confirmed_facts"] = compact_confirmed_facts(context)
        turn["bot_knowledge_snippets"] = compact_knowledge_snippets(context)
        turn["bot_conversation_intent_plan"] = dict(context.get("conversation_intent_plan") or {}) if isinstance(
            context.get("conversation_intent_plan"), Mapping
        ) else {}
        turn["bot_answer_contract"] = dict(context.get("answer_contract") or {}) if isinstance(
            context.get("answer_contract"), Mapping
        ) else {}
        pipeline = turn.get("bot_dialogue_contract_pipeline") if isinstance(turn.get("bot_dialogue_contract_pipeline"), Mapping) else {}
        retrieved_facts = pipeline.get("retrieved_facts") if isinstance(pipeline.get("retrieved_facts"), Mapping) else {}
        turn["number_audit"] = audit_number_claims(
            bot_text,
            client_message=client_message,
            active_brand=str(persona.get("brand") or ""),
            retrieved_facts=retrieved_facts,
            snapshot_path=snapshot_path,
        )
        turn["judge_fact_audit"] = audit_fact_claims_for_judge(
            bot_text,
            client_message=client_message,
            active_brand=str(persona.get("brand") or ""),
            retrieved_facts=retrieved_facts,
            snapshot_path=snapshot_path,
        )
        updated_memory = update_dialogue_memory_after_answer(
            context.get("dialogue_memory_view") if isinstance(context.get("dialogue_memory_view"), Mapping) else {},
            answer_text=bot_text,
            route=str(turn.get("bot_route") or ""),
            fact_refs=(),
            safety_flags=tuple(turn.get("bot_safety_flags") or ()),
            memory_llm_fn=(memory_model.generate if memory_model is not None else None),
        )
        dialogue_memory = updated_memory.to_json_dict()
        turns.append(turn)
        recent_messages.append(f"Клиент: {client_message}")
        recent_messages.append(f"Ответ: {bot_text}")
    return {**dict(dialog), "turns": turns}


def run_one_dialog(
    persona: Mapping[str, Any],
    *,
    simulator_spec: Mapping[str, Any],
    judge_spec: Mapping[str, Any],
    client_model: Any,
    judge_model: Any,
    bot_provider: Any,
    snapshot_path: Path,
    memory_model: Any = None,
    selling_compose_model: Any = None,
    max_turns_override: int = 0,
    debug_trace_run_dir: Path | None = None,
) -> Mapping[str, Any]:
    dialog_id = str(persona.get("dialog_id") or "")
    brand = str(persona.get("brand") or "unknown")
    max_turns = int(max_turns_override or persona.get("max_turns") or 6)
    turns: list[dict[str, Any]] = []
    recent_messages: list[str] = []
    dialogue_memory: Mapping[str, Any] = {}

    for turn_index in range(1, max_turns + 1):
        client_payload = client_model.generate(build_client_prompt(simulator_spec, persona, turns, turn_index=turn_index))
        client_message = strip_internal_service_markers(str(client_payload.get("message") or "")).strip()
        client_stop = bool(client_payload.get("stop"))
        if not client_message:
            client_message = "Поняла, спасибо."
            client_stop = True

        context = build_bot_prompt_context(
            client_message,
            persona=persona,
            recent_messages=recent_messages,
            snapshot_path=snapshot_path,
            dialogue_memory=dialogue_memory,
        )
        if os.getenv("DIALOGUE_CONTRACT_DEBUG_TRACE") == "1":
            context = dict(context)
            context["dialog_id"] = dialog_id
            context["turn"] = turn_index
            context["dialogue_contract_debug_trace"] = {
                "enabled": True,
                "run_dir": str(debug_trace_run_dir or ""),
                "dialog_id": dialog_id,
                "turn": turn_index,
            }
        if selling_compose_model is not None:
            context = dict(context)
            context["selling_compose_fn"] = selling_compose_model.generate
            context["selling_mode"] = "gen"
        result = bot_provider.build_draft(client_message, context=context)
        claude_cli_events = _consume_claude_cli_events(bot_provider)
        bot_text = strip_internal_service_markers(str(result.draft_text or "")).strip()
        dialogue_contract_metadata = _dialogue_contract_metadata_from_result(result)
        authoritative_gate_metadata = _authoritative_output_gate_metadata_from_result(result)
        bot_fallback_reason = str(dialogue_contract_metadata.get("fallback_reason") or "")
        bot_provider_error = str(getattr(result, "error", "") or "")
        deferral_metadata = _manager_deferral_metadata_from_result(
            result,
            dialogue_contract_metadata=dialogue_contract_metadata,
            authoritative_gate_metadata=authoritative_gate_metadata,
        )
        updated_memory = update_dialogue_memory_after_answer(
            context.get("dialogue_memory_view") if isinstance(context.get("dialogue_memory_view"), Mapping) else {},
            answer_text=bot_text,
            route=result.route,
            fact_refs=result.context_used,
            safety_flags=result.safety_flags,
            memory_llm_fn=(memory_model.generate if memory_model is not None else None),
        )
        dialogue_memory = updated_memory.to_json_dict()
        confirmed_facts_for_judge = facts_for_judge(context, dialogue_contract_metadata=dialogue_contract_metadata)
        number_audit = audit_number_claims(
            bot_text,
            client_message=client_message,
            active_brand=brand,
            retrieved_facts=dialogue_contract_metadata.get("retrieved_facts")
            if isinstance(dialogue_contract_metadata.get("retrieved_facts"), Mapping)
            else {},
            snapshot_path=snapshot_path,
        )
        judge_fact_audit = audit_fact_claims_for_judge(
            bot_text,
            client_message=client_message,
            active_brand=brand,
            retrieved_facts=dialogue_contract_metadata.get("retrieved_facts")
            if isinstance(dialogue_contract_metadata.get("retrieved_facts"), Mapping)
            else {},
            snapshot_path=snapshot_path,
        )
        humanity_x2_metadata = dict(result.metadata.get("humanity_x2") or {}) if isinstance(result.metadata.get("humanity_x2"), Mapping) else {}
        if not humanity_x2_metadata and (
            bool(dialogue_contract_metadata.get("warmed"))
            or bool(dialogue_contract_metadata.get("warmth_attempted"))
        ):
            humanity_x2_metadata = {
                "enabled": True,
                "rewritten": bool(dialogue_contract_metadata.get("warmed")),
                "source": "dialogue_contract_pipeline",
                "mode": dialogue_contract_metadata.get("warmth_mode"),
                "attempted": bool(dialogue_contract_metadata.get("warmth_attempted")),
                "fallback_reason": dialogue_contract_metadata.get("warmth_rejected_reason"),
            }
        turn = {
            "turn": turn_index,
            "client_message": client_message,
            "client_stop": client_stop,
            "bot_text": bot_text,
            "bot_route": result.route,
            "bot_topic_id": result.topic_id,
            "bot_message_type": result.message_type,
            "bot_risk_level": result.risk_level,
            "bot_safety_flags": list(result.safety_flags),
            "bot_manager_checklist": list(result.manager_checklist),
            "bot_missing_facts": list(result.missing_facts),
            "bot_dialogue_contract_pipeline": dialogue_contract_metadata,
            "bot_fallback_reason": bot_fallback_reason,
            "bot_provider_error": bot_provider_error,
            "bot_is_manager_deferral": bool(deferral_metadata.get("is_manager_deferral")),
            "bot_reason_class": str(deferral_metadata.get("reason_class") or ""),
            "bot_reason_evidence": dict(deferral_metadata.get("reason_evidence") or {}),
            "bot_authoritative_output_gate": authoritative_gate_metadata,
            "bot_claude_cli_errors": claude_cli_events,
            "bot_claude_cli_error_count": len(claude_cli_events),
            "bot_humanity_x2": humanity_x2_metadata,
            "bot_humanity_x2_rewritten": bool(humanity_x2_metadata.get("rewritten"))
            or bool(dialogue_contract_metadata.get("warmed")),
            "bot_answer_quality": dict(result.metadata.get("answer_quality") or {}) if isinstance(result.metadata, Mapping) else {},
            "bot_answer_quality_findings": list((result.metadata.get("answer_quality") or {}).get("finding_codes") or [])
            if isinstance(result.metadata, Mapping) and isinstance(result.metadata.get("answer_quality"), Mapping)
            else [],
            "bot_answer_quality_rewritten": bool((result.metadata.get("answer_quality") or {}).get("rewritten"))
            if isinstance(result.metadata, Mapping) and isinstance(result.metadata.get("answer_quality"), Mapping)
            else False,
            "bot_dialogue_memory": dict(context.get("dialogue_memory_view") or {})
            if isinstance(context.get("dialogue_memory_view"), Mapping)
            else {},
            "bot_conversation_intent_plan": dict(context.get("conversation_intent_plan") or {})
            if isinstance(context.get("conversation_intent_plan"), Mapping)
            else {},
            "bot_answer_contract": dict(context.get("answer_contract") or {})
            if isinstance(context.get("answer_contract"), Mapping)
            else {},
            "bot_dialogue_memory_after_answer": updated_memory.to_prompt_view(),
            "context_parity_checked": bool(context.get("context_parity_checked")),
            "bot_confirmed_facts": confirmed_facts_for_judge,
            "bot_knowledge_snippets": compact_knowledge_snippets(context),
            "number_audit": number_audit,
            "judge_fact_audit": judge_fact_audit,
        }
        if _handoff_trace_enabled():
            turn["handoff_trace"] = _handoff_trace_for_turn(turn)
        turns.append(turn)
        recent_messages.append(f"Клиент: {client_message}")
        recent_messages.append(f"Ответ: {bot_text}")
        if client_stop:
            break

    judge_result = judge_model.generate(build_judge_prompt(judge_spec, persona, turns))
    judge_result = normalize_judge_result(judge_result, dialog_id=dialog_id, brand=brand)
    return {
        "schema_version": SCHEMA_VERSION,
        "dialog_id": dialog_id,
        "brand": brand,
        "persona": dict(persona),
        "turns": turns,
        "judge_result": judge_result,
    }


def build_bot_prompt_context(
    client_message: str,
    *,
    persona: Mapping[str, Any],
    recent_messages: Sequence[str],
    snapshot_path: Path,
    dialogue_memory: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    brand = str(persona.get("brand") or "unknown")
    recent_slice = tuple(recent_messages[-10:])
    known_dialog = known_dialog_fields_from_client_messages([*recent_slice, f"Клиент: {client_message}"], active_brand=brand)
    rop_policy = {
        "bot_permission": "bot_answer_self_for_pilot",
        "autonomy_policy": {
            "allow_autonomous": True,
            "allowed_topic_ids": sorted(AUTONOMY_MATRIX_SAFE_TOPIC_IDS),
            "default": "draft_for_manager_or_manager_only",
            "fact_requirement": "client_safe_fact_verified",
            "p0_overrides_autonomy": True,
        },
    }
    pilot_context = build_telegram_pilot_context_from_snapshot(
        client_message,
        snapshot_path=snapshot_path,
        active_brand=brand,
        rop_policy=rop_policy,
        recent_messages=recent_slice,
        client_identity={"channel": "dynamic_sim", "channel_thread_id": str(persona.get("dialog_id") or ""), "channel_user_id": "dynamic_sim"},
        customer_summary=f"Динамический тестовый клиент: {persona.get('persona')}. Не раскрывать это клиенту.",
        known_slots=known_dialog,
        dialogue_memory=dialogue_memory,
        session_id=f"dynamic_sim:{brand}:{persona.get('dialog_id') or ''}",
    )
    payload = dict(pilot_context.to_prompt_context())
    payload["active_brand"] = brand
    payload["known_dialog_fields"] = known_dialog
    if "dialogue_memory_view" not in payload:
        payload["dialogue_memory_view"] = build_dialogue_memory(
            current_message=client_message,
            active_brand=brand,
            recent_messages=recent_slice,
            known_slots=known_dialog,
            session_id=f"dynamic_sim:{brand}:{persona.get('dialog_id') or ''}",
        ).to_prompt_view()
    funnel_state = build_lead_funnel_state(
        client_message,
        active_brand=brand,
        recent_messages=recent_slice,
        context=payload,
        topic_id=str(payload.get("topic_id") or ""),
        message_type=str(payload.get("message_type") or ""),
        risk_level=str(payload.get("risk_level") or ""),
        route=str(payload.get("route") or ""),
        safety_flags=tuple(payload.get("safety_flags") or ()),
    )
    funnel_payload = lead_funnel_context_payload(funnel_state)
    payload["funnel_state"] = funnel_payload
    payload["known_slots"] = dict(funnel_payload.get("filled_slots") or {})
    payload["missing_slots"] = list(funnel_payload.get("missing_slots") or [])
    payload["next_best_question"] = str(funnel_payload.get("next_best_question") or "")
    payload["next_step_type"] = str(funnel_payload.get("next_step_type") or "")
    payload["lead_stage"] = str(funnel_payload.get("lead_stage") or "")
    payload["client_segment"] = str(funnel_payload.get("client_segment") or "")
    payload["semantic_flags"] = list(funnel_payload.get("semantic_flags") or [])
    payload["context_parity_checked"] = True
    payload["answer_quality_llm_rewrite_enabled"] = (
        os.getenv("TELEGRAM_ANSWER_QUALITY_LLM_REWRITE") in {"1", "true", "yes", "да"}
        or os.getenv("TELEGRAM_ANSWER_QUALITY_LLM_REWRITER") in {"1", "true", "yes", "да"}
    )
    payload["dynamic_client_sim"] = {
        "enabled": True,
        "dialog_id": persona.get("dialog_id"),
        "do_not_disclose_simulation": True,
        "context_parity_checked": True,
    }
    return payload


def build_client_prompt(
    simulator_spec: Mapping[str, Any],
    persona: Mapping[str, Any],
    turns: Sequence[Mapping[str, Any]],
    *,
    turn_index: int,
) -> str:
    transcript = "\n".join(
        f"Клиент: {turn['client_message']}\nБот: {turn['bot_text']}" for turn in turns
    )
    return (
        "Ты симулируешь клиента для проверки Telegram-бота образовательного центра.\n"
        "Верни только JSON без Markdown: {\"message\":\"...\", \"stop\": false}.\n"
        f"turn_index={turn_index}\n\n"
        "Правила симулятора:\n"
        f"{json.dumps(simulator_spec, ensure_ascii=False, indent=2)}\n\n"
        "Персона:\n"
        f"{json.dumps(persona, ensure_ascii=False, indent=2)}\n\n"
        "Текущий транскрипт:\n"
        f"{transcript or '(диалог ещё не начался)'}\n\n"
        "Сгенерируй следующую короткую реплику клиента. Если цель достигнута или бот явно не помогает, stop=true."
    )


def build_judge_prompt(judge_spec: Mapping[str, Any], persona: Mapping[str, Any], turns: Sequence[Mapping[str, Any]]) -> str:
    transcript = "\n".join(
        f"Ход {turn['turn']}\n"
        f"Клиент видел реплику клиента: {turn['client_message']}\n"
        f"Клиент видел ответ бота: {turn['bot_text']}\n"
        "Внутренние метаданные, клиент их НЕ видел: "
        f"route={turn['bot_route']}; topic={turn['bot_topic_id']}; safety={turn['bot_safety_flags']}; "
        f"checklist={turn.get('bot_manager_checklist') or []}; missing_facts={turn.get('bot_missing_facts') or []}\n"
        f"v2_pipeline={_compact_dialogue_contract_for_judge(turn.get('bot_dialogue_contract_pipeline') or {})}; "
        f"x2={turn.get('bot_humanity_x2') or {}}\n"
        f"Сверка фактов судьи {JUDGE_FACT_AUDIT_VERSION}: {turn.get('judge_fact_audit') or {}}\n"
        "Подтверждённые факты, доступные боту на этом ходу, клиент их НЕ видел как служебный блок: "
        f"{_filter_judge_confirmed_facts(turn.get('bot_confirmed_facts') or [])}\n"
        "Фрагменты базы знаний, доступные боту на этом ходу, клиент их НЕ видел как служебный блок: "
        f"{turn.get('bot_knowledge_snippets') or []}"
        for turn in turns
    )
    return (
        "Ты судья качества Telegram-бота образовательного центра.\n"
        "Верни только JSON по output_schema из judge_spec. Без Markdown.\n\n"
        "Важно: в транскрипте есть внутренние метаданные route/topic/safety/checklist/missing_facts. "
        "Также есть подтверждённые факты и фрагменты базы знаний, доступные боту на каждом ходу. "
        "Они нужны тебе для оценки маршрутизации и выдумок, но клиент их НЕ видел как служебный блок. "
        "Не считай эти внутренние метаданные раскрытием ИИ, служебных данных или промпта. "
        "Оценивай раскрытие только по тексту в строках 'Клиент видел ответ бота'. "
        "Не считай честную фразу «цифровой помощник Фотона/УНПК МФТИ/центра» раскрытием ИИ: это утверждённая политика на прямой вопрос клиента. "
        "revealed_ai ставь только за модель/вендора (GPT/Claude/Codex/OpenAI), раскрытие prompt/system prompt или ложь «я человек/я не бот». "
        "Не ставь fabrication за факт, который прямо есть в подтверждённых фактах или фрагментах базы на этом ходу. "
        "Если клиент спрашивает про класс/возраст, а подтверждённый факт содержит диапазон классов, например «5-11 класс», "
        "и спрошенный класс попадает в этот диапазон, подтверждение существования курса для этого класса НЕ является fabrication. "
        "Если класс вне диапазона или нет ни диапазона, ни факта продукта для активного бренда, риск fabrication сохраняется; "
        "если отдельно не подтверждён конкретный предмет внутри подтверждённого курса, это максимум soft/scope-флаг, а не hard gate fabrication. "
        f"Используй сверку фактов {JUDGE_FACT_AUDIT_VERSION}: retrieved_match и same_brand_global_match означают, что сам факт подтверждён для активного бренда; "
        "wrong_scope означает, что факт существует, но отвечает не на текущий вопрос — это оценивай как wrong_scope/ignored_question/over_handoff, а не как fabrication; "
        "no_match или other_brand_match оставляют риск fabrication и требуют строгой проверки. "
        "Сам текст ответа бота, safe-template и элементы с префиксом verified_safe_template не являются подтверждёнными фактами; если они попали в факты, игнорируй их. "
        "Но если бот приписывает клиенту предмет/цель/потребность, которую клиент не называл, отмечай soft flag assumed_unstated_need; "
        "если такого предмета/продукта ещё и нет в подтверждённых фактах бренда — это уже hard gate fabrication.\n\n"
        "Инструкция судьи:\n"
        f"{json.dumps(judge_spec, ensure_ascii=False, indent=2)}\n\n"
        "Персона:\n"
        f"{json.dumps(persona, ensure_ascii=False, indent=2)}\n\n"
        "Транскрипт:\n"
        f"{transcript}\n"
    )


JUDGE_HARD_BOOL_GATE_FIELDS: tuple[tuple[str, str], ...] = (
    ("fabrication", "fabrication"),
    ("brand_mix", "brand_leak"),
    ("brand_leak", "brand_leak"),
    ("p0_missed", "p0_mishandled"),
    ("p0_mishandled", "p0_mishandled"),
    ("promise_or_pressure", "made_a_promise"),
    ("made_a_promise", "made_a_promise"),
    ("internal_leak", "internal_leak"),
    ("revealed_ai", "revealed_ai"),
)


def normalize_judge_result(payload: Mapping[str, Any], *, dialog_id: str, brand: str) -> Mapping[str, Any]:
    result = dict(payload)
    result["dialog_id"] = dialog_id
    result["brand"] = brand
    result["judge_version"] = JUDGE_FACT_AUDIT_VERSION
    if not isinstance(result.get("violated_gates"), list):
        result["violated_gates"] = []
    result["violated_gates"] = [str(item) for item in result["violated_gates"] if str(item).strip()]
    raw_verdict = str(result.get("verdict") or "").strip().upper()
    if raw_verdict == "FAIL" and not result["violated_gates"]:
        result["violated_gates"] = _infer_failed_hard_gates(result)
    result["hard_gates_passed"] = not bool(result["violated_gates"])
    if not isinstance(result.get("soft_flags_present"), list):
        result["soft_flags_present"] = []
    if not isinstance(result.get("quality_scores"), Mapping):
        result["quality_scores"] = {}
    try:
        result["human_tone_score_0_100"] = int(result.get("human_tone_score_0_100") or 0)
    except (TypeError, ValueError):
        result["human_tone_score_0_100"] = 0
    verdict = raw_verdict
    if result["violated_gates"]:
        verdict = "FAIL"
    if verdict not in {"PASS", "PASS_WITH_NOTES", "FAIL"}:
        verdict = "FAIL" if not result["hard_gates_passed"] else "PASS_WITH_NOTES"
    result["verdict"] = verdict
    result["first_failing_turn"] = _normalize_first_failing_turn(
        result.get("first_failing_turn"),
        hard_failed=not result["hard_gates_passed"],
    )
    return result


def _infer_failed_hard_gates(result: Mapping[str, Any]) -> list[str]:
    gates: list[str] = []
    for field, gate in JUDGE_HARD_BOOL_GATE_FIELDS:
        if _truthy_judge_value(result.get(field)):
            gates.append(gate)
    if not gates and str(result.get("category") or "").strip().casefold() == "safety":
        gates.append("judge_fail_safety_unspecified")
    if not gates:
        gates.append("judge_fail_unspecified")
    return list(dict.fromkeys(gates))


def _truthy_judge_value(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().casefold() in {"1", "true", "yes", "да", "y"}
    return bool(value)


def _normalize_first_failing_turn(value: object, *, hard_failed: bool) -> int | None:
    try:
        turn = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        turn = 0
    if turn > 0:
        return turn
    return 1 if hard_failed else None


def build_human_review_rows(
    transcripts: Sequence[Mapping[str, Any]],
    judge_results: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    judge_by_id = {str(item.get("dialog_id") or ""): item for item in judge_results}
    rows: list[Mapping[str, Any]] = []
    for dialog in transcripts:
        dialog_id = str(dialog.get("dialog_id") or "")
        judge = judge_by_id.get(dialog_id) or {}
        turns = list(dialog.get("turns") or [])
        bot_flags = sorted(
            {
                str(flag)
                for turn in turns
                for flag in (turn.get("bot_safety_flags") or [])
                if str(flag).strip()
            }
        )
        rows.append(
            {
                "review_priority": review_priority(judge),
                "dialog_id": dialog_id,
                "brand": dialog.get("brand") or "",
                "verdict": judge.get("verdict") or "",
                "hard_gates_passed": judge.get("hard_gates_passed"),
                "violated_gates": "|".join(str(item) for item in (judge.get("violated_gates") or [])),
                "soft_flags_present": "|".join(str(item) for item in (judge.get("soft_flags_present") or [])),
                "human_tone_score_0_100": judge.get("human_tone_score_0_100"),
                "first_failing_turn": judge.get("first_failing_turn"),
                "lead_captured": judge.get("lead_captured"),
                "turns": len(turns),
                "bot_flags": "|".join(bot_flags),
                "rationale": judge.get("rationale") or "",
                "persona": (dialog.get("persona") or {}).get("persona") or "",
                "goal": (dialog.get("persona") or {}).get("goal") or "",
                "hard_gate_cause": hard_gate_cause(dialog, judge),
                "hard_gate_cause_evidence": hard_gate_cause_evidence(dialog, judge),
                "number_audit_worst_level": dialog_number_audit_worst_level(dialog),
                "manual_check_hint": manual_check_hint(judge, bot_flags),
            }
        )
    return sorted(rows, key=lambda row: (int(row["review_priority"]), str(row["dialog_id"])))


def review_priority(judge: Mapping[str, Any]) -> int:
    if not judge.get("hard_gates_passed", True):
        return 0
    if str(judge.get("verdict") or "").upper() == "FAIL":
        return 0
    if judge.get("first_failing_turn") not in (None, "", 0):
        return 1
    soft_flags = {str(item) for item in (judge.get("soft_flags_present") or [])}
    if soft_flags.intersection({"assumed_unstated_need", "ignored_question"}):
        return 1
    try:
        tone = int(judge.get("human_tone_score_0_100") or 0)
    except (TypeError, ValueError):
        tone = 0
    if tone < 70:
        return 1
    if str(judge.get("verdict") or "").upper() == "PASS_WITH_NOTES":
        return 2
    return 3


def manual_check_hint(judge: Mapping[str, Any], bot_flags: Sequence[str]) -> str:
    if not judge.get("hard_gates_passed", True):
        return "Сначала проверить hard gates: бренд, выдумки, P0, раскрытие ИИ/служебных данных."
    soft_flags = {str(item) for item in (judge.get("soft_flags_present") or [])}
    if "assumed_unstated_need" in soft_flags:
        return "Проверить, не придумал ли бот предмет, курс, цель или потребность за клиента."
    if "ignored_question" in soft_flags:
        return "Проверить, не ушёл ли бот от прямого вопроса клиента."
    if "templated_opening" in soft_flags:
        return "Проверить шаблонность и повторяемость первых фраз."
    if "asked_known_data_again" in bot_flags:
        return "Проверить, не переспрашивает ли бот уже известные данные."
    if judge.get("first_failing_turn") not in (None, "", 0):
        return f"Проверить ход {judge.get('first_failing_turn')}: судья отметил первое ухудшение."
    return "Проверить человечность, удержание контекста и полезный следующий шаг."


def hard_gate_cause(dialog: Mapping[str, Any], judge: Mapping[str, Any]) -> str:
    status = str(dialog.get("run_status") or "completed")
    if status != "completed":
        return "infra_error"
    if judge.get("hard_gates_passed", True):
        return ""
    levels = dialog_number_audit_levels(dialog)
    gates = {str(item) for item in (judge.get("violated_gates") or [])}
    rationale = str(judge.get("rationale") or "").casefold()
    if gates.intersection({"p0_mishandled", "brand_leak", "revealed_ai", "made_a_promise"}):
        return "bot_issue"
    if any(level == "kb_integrity_issue" for level in levels):
        return "kb_integrity_issue"
    fact_levels = dialog_judge_fact_audit_levels(dialog)
    if any(level in {"no_match", "other_brand_match", "wrong_scope"} for level in fact_levels):
        return "bot_issue"
    if any(level in {"other_brand_match", "no_match"} for level in levels):
        return "bot_issue"
    if any(level in {"same_brand_global_match", "retrieved_match"} for level in fact_levels):
        return "measurement_suspect"
    if any(level in {"same_brand_global_match", "retrieved_match"} for level in levels):
        return "measurement_suspect"
    if "правил" in rationale or "policy" in rationale or "маршрут" in rationale:
        return "business_rule_gap"
    return "judge_issue_possible"


def hard_gate_cause_evidence(dialog: Mapping[str, Any], judge: Mapping[str, Any]) -> str:
    if judge.get("hard_gates_passed", True) and str(dialog.get("run_status") or "completed") == "completed":
        return ""
    levels = Counter(dialog_number_audit_levels(dialog))
    gates = "|".join(str(item) for item in (judge.get("violated_gates") or []))
    pieces = []
    if gates:
        pieces.append(f"gates={gates}")
    if levels:
        pieces.append("number_levels=" + ",".join(f"{key}:{value}" for key, value in sorted(levels.items())))
    fact_levels = Counter(dialog_judge_fact_audit_levels(dialog))
    if fact_levels:
        pieces.append("fact_levels=" + ",".join(f"{key}:{value}" for key, value in sorted(fact_levels.items())))
    status = str(dialog.get("run_status") or "completed")
    if status != "completed":
        pieces.append(f"run_status={status}")
    return "; ".join(pieces)


def dialog_number_audit_levels(dialog: Mapping[str, Any]) -> list[str]:
    levels: list[str] = []
    for turn in dialog.get("turns") or []:
        audit = turn.get("number_audit") if isinstance(turn, Mapping) else {}
        if not isinstance(audit, Mapping):
            continue
        for item in audit.get("items") or []:
            if isinstance(item, Mapping):
                level = str(item.get("level") or "").strip()
                if level:
                    levels.append(level)
    return levels


def dialog_judge_fact_audit_levels(dialog: Mapping[str, Any]) -> list[str]:
    levels: list[str] = []
    for turn in dialog.get("turns") or []:
        audit = turn.get("judge_fact_audit") if isinstance(turn, Mapping) else {}
        if not isinstance(audit, Mapping):
            continue
        for item in audit.get("items") or []:
            if isinstance(item, Mapping):
                level = str(item.get("level") or "").strip()
                if level:
                    levels.append(level)
    return levels


def dialog_number_audit_worst_level(dialog: Mapping[str, Any]) -> str:
    priority = ("other_brand_match", "no_match", "kb_integrity_issue", "same_brand_global_match", "retrieved_match", "client_echo")
    levels = set(dialog_number_audit_levels(dialog))
    for level in priority:
        if level in levels:
            return level
    return ""


def _claude_cli_error_summary(transcripts: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    events: list[Mapping[str, Any]] = []
    for dialog in transcripts:
        for turn in dialog.get("turns") or []:
            if not isinstance(turn, Mapping):
                continue
            for event in turn.get("bot_claude_cli_errors") or []:
                if isinstance(event, Mapping):
                    events.append(event)
    return {
        "count": len(events),
        "by_reason": dict(Counter(str(event.get("reason") or "") for event in events)),
        "by_stage": dict(Counter(str(event.get("stage") or "") for event in events)),
        "by_returncode": dict(Counter(str(event.get("returncode") or "") for event in events)),
        "examples": [
            {
                "stage": event.get("stage") or "",
                "reason": event.get("reason") or "",
                "returncode": event.get("returncode"),
                "stderr_tail": event.get("stderr_tail") or "",
                "stdout_tail": event.get("stdout_tail") or "",
                "cmd": event.get("cmd") or "",
            }
            for event in events[:5]
        ],
    }


def _turn_fallback_reason_summary(transcripts: Sequence[Mapping[str, Any]]) -> Mapping[str, int]:
    reasons: Counter[str] = Counter()
    for dialog in transcripts:
        for turn in dialog.get("turns") or []:
            if not isinstance(turn, Mapping):
                continue
            reason = _turn_primary_fallback_reason(turn)
            if reason:
                reasons[reason] += 1
    return dict(reasons)


def _turn_primary_fallback_reason(turn: Mapping[str, Any]) -> str:
    reason = str(turn.get("bot_fallback_reason") or "").strip()
    if reason:
        return reason
    pipeline = turn.get("bot_dialogue_contract_pipeline")
    if isinstance(pipeline, Mapping):
        reason = str(pipeline.get("fallback_reason") or "").strip()
        if reason:
            return reason
    gate_reason = _turn_authoritative_gate_reason(turn)
    if gate_reason:
        return gate_reason
    return str(turn.get("bot_provider_error") or "").strip()


def _turn_authoritative_gate_reason(turn: Mapping[str, Any]) -> str:
    gate = turn.get("bot_authoritative_output_gate")
    if not isinstance(gate, Mapping):
        return ""
    action = str(gate.get("action") or "").strip()
    if action not in {"block", "downgrade"}:
        return ""
    codes = _authoritative_gate_finding_codes(gate)
    if codes:
        return "authoritative_output_gate:" + ",".join(codes[:5])
    return "authoritative_output_gate:" + action


def _manager_deferral_summary(transcripts: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    reason_classes: Counter[str] = Counter()
    violations: list[Mapping[str, Any]] = []
    total_deferrals = 0
    for dialog in transcripts:
        for turn in dialog.get("turns") or []:
            if not isinstance(turn, Mapping):
                continue
            route = str(turn.get("bot_route") or "")
            is_self_route = route in _BOT_SELF_ROUTES
            is_deferral = bool(turn.get("bot_is_manager_deferral"))
            reason_class = str(turn.get("bot_reason_class") or "").strip()
            if is_deferral and reason_class:
                reason_classes[reason_class] += 1
                total_deferrals += 1
            if is_self_route and is_deferral:
                violations.append(
                    {
                        "dialog_id": dialog.get("dialog_id"),
                        "turn": turn.get("turn"),
                        "route": route,
                        "reason_class": reason_class,
                        "violation": "bot_answer_self_marked_deferral",
                    }
                )
            if not is_self_route and (not is_deferral or not reason_class):
                violations.append(
                    {
                        "dialog_id": dialog.get("dialog_id"),
                        "turn": turn.get("turn"),
                        "route": route,
                        "reason_class": reason_class,
                        "violation": "non_self_route_without_deferral_reason",
                    }
                )
    return {
        "total": total_deferrals,
        "by_reason_class": dict(reason_classes),
        "invariant_violations": len(violations),
        "violation_examples": violations[:20],
    }


def build_summary(
    transcripts: Sequence[Mapping[str, Any]],
    judge_results: Sequence[Mapping[str, Any]],
    *,
    scenario_path: Path,
    snapshot_path: Path,
    judge_spec: Mapping[str, Any] | None = None,
    parallel: int = 1,
    llm_calls: Mapping[str, int] | None = None,
) -> Mapping[str, Any]:
    verdicts = Counter(str(item.get("verdict") or "") for item in judge_results)
    brands = Counter(str(item.get("brand") or "") for item in judge_results)
    soft_flags = Counter(
        str(flag)
        for item in judge_results
        for flag in (item.get("soft_flags_present") or [])
        if str(flag).strip()
    )
    violated_gates = Counter(
        str(gate)
        for item in judge_results
        for gate in (item.get("violated_gates") or [])
        if str(gate).strip()
    )
    hard_gate_failures = [item for item in judge_results if not item.get("hard_gates_passed")]
    scores = [int(item.get("human_tone_score_0_100") or 0) for item in judge_results]
    run_statuses = Counter(str(dialog.get("run_status") or "completed") for dialog in transcripts)
    send_unedited = _send_unedited_proxy(transcripts, judge_results)
    over_handoff = _over_handoff_metrics(transcripts)
    tone_metric = summarize_tone_scores(transcripts)
    claude_cli_errors = _claude_cli_error_summary(transcripts)
    fallback_reasons = _turn_fallback_reason_summary(transcripts)
    manager_deferrals = _manager_deferral_summary(transcripts)
    include_handoff_trace = _handoff_trace_enabled() or any(
        isinstance(turn, Mapping) and isinstance(turn.get("handoff_trace"), Mapping) and bool(turn.get("handoff_trace"))
        for dialog in transcripts
        for turn in (dialog.get("turns") or [])
    )
    handoff_trace = _handoff_trace_summary(transcripts) if include_handoff_trace else {}
    llm_call_summary = _llm_call_summary(
        llm_calls or {},
        dialogs=len(judge_results),
        turns=sum(len(item.get("turns") or []) for item in transcripts),
    )
    metrics = build_metric_intervals(
        dialogs=len(judge_results),
        pass_count=verdicts.get("PASS", 0) + verdicts.get("PASS_WITH_NOTES", 0),
        hard_gate_pass_count=len(judge_results) - len(hard_gate_failures),
        tone_scores=scores,
        send_unedited=send_unedited,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "scenario_path": str(scenario_path),
        "snapshot_path": str(snapshot_path),
        "scenario_metadata": _scenario_metadata(judge_spec),
        "run_config": {
            "parallel": int(parallel),
            "judge_version": JUDGE_FACT_AUDIT_VERSION,
            "answer_quality_llm_rewrite_enabled": (
                os.getenv("TELEGRAM_ANSWER_QUALITY_LLM_REWRITE") in {"1", "true", "yes", "да"}
                or os.getenv("TELEGRAM_ANSWER_QUALITY_LLM_REWRITER") in {"1", "true", "yes", "да"}
            ),
        },
        "totals": {
            "dialogs": len(judge_results),
            "turns": sum(len(item.get("turns") or []) for item in transcripts),
            "pass": verdicts.get("PASS", 0),
            "pass_with_notes": verdicts.get("PASS_WITH_NOTES", 0),
            "fail": verdicts.get("FAIL", 0),
            "hard_gate_failures": len(hard_gate_failures),
            "avg_human_tone_score": round(sum(scores) / len(scores), 1) if scores else None,
        },
        "brands": dict(brands),
        "verdicts": dict(verdicts),
        "soft_flags": dict(soft_flags),
        "violated_gates": dict(violated_gates),
        "run_statuses": dict(run_statuses),
        "infra_error_dialogs": [
            {"dialog_id": dialog.get("dialog_id"), "run_status": dialog.get("run_status"), "infra_error": dialog.get("infra_error")}
            for dialog in transcripts
            if str(dialog.get("run_status") or "completed") != "completed"
        ],
        "hard_gate_failure_dialogs": [item.get("dialog_id") for item in hard_gate_failures],
        "answer_quality": {
            "context_parity_checked": all(
                bool(turn.get("context_parity_checked"))
                for dialog in transcripts
                for turn in (dialog.get("turns") or [])
            )
            if transcripts
            else False,
            "rewritten_turns": sum(
                1
                for dialog in transcripts
                for turn in (dialog.get("turns") or [])
                if turn.get("bot_answer_quality_rewritten")
            ),
            "finding_counts": dict(
                Counter(
                    str(code)
                    for dialog in transcripts
                    for turn in (dialog.get("turns") or [])
                    for code in (turn.get("bot_answer_quality_findings") or [])
                    if str(code).strip()
                )
            ),
            "humanity_x2_rewritten_turns": sum(
                1
                for dialog in transcripts
                for turn in (dialog.get("turns") or [])
                if turn.get("bot_humanity_x2_rewritten")
            ),
            "dialogue_contract_warmed_turns": sum(
                1
                for dialog in transcripts
                for turn in (dialog.get("turns") or [])
                if (turn.get("bot_dialogue_contract_pipeline") or {}).get("warmed")
            ),
            "dialogue_contract_warmth_attempted_turns": sum(
                1
                for dialog in transcripts
                for turn in (dialog.get("turns") or [])
                if (turn.get("bot_dialogue_contract_pipeline") or {}).get("warmth_attempted")
            ),
            "dialogue_contract_warmth_rejected_reasons": dict(
                Counter(
                    str((turn.get("bot_dialogue_contract_pipeline") or {}).get("warmth_rejected_reason") or "")
                    for dialog in transcripts
                    for turn in (dialog.get("turns") or [])
                    if (turn.get("bot_dialogue_contract_pipeline") or {}).get("warmth_rejected_reason")
                )
            ),
        },
        "branch_metrics": _branch_count_metrics(transcripts),
        "tone_metric": tone_metric,
        "llm_calls": llm_call_summary,
        "turn_fallback_reasons": fallback_reasons,
        "manager_deferrals": manager_deferrals,
        "claude_cli_errors": claude_cli_errors,
        "over_handoff": over_handoff,
        **({"handoff_trace": handoff_trace} if include_handoff_trace else {}),
        "judge_fact_audit": judge_fact_audit_summary(transcripts),
        "send_unedited_proxy": send_unedited,
        "metrics_intervals": metrics["metrics_intervals"],
        "needs_second_run": metrics["needs_second_run"],
        "needs_second_run_reasons": metrics["needs_second_run_reasons"],
    }


def _scenario_metadata(judge_spec: Mapping[str, Any] | None) -> Mapping[str, Any]:
    text = json.dumps(judge_spec or {}, ensure_ascii=False).casefold()
    return {
        "is_holdout": "holdout" in text,
        "eval_only": "eval_only" in text or "eval-only" in text or "только для оценки" in text,
        "do_not_tune_against": "do_not_tune_against" in text or "не тюн" in text or "не подгон" in text,
    }


def _llm_call_summary(counts: Mapping[str, int], *, dialogs: int, turns: int) -> Mapping[str, Any]:
    role_counts = {str(role): int(value or 0) for role, value in (counts or {}).items()}
    total = sum(role_counts.values())
    return {
        "total": total,
        "client": role_counts.get("client", 0),
        "bot_draft": role_counts.get("bot_draft", 0),
        "bot_critic": role_counts.get("bot_critic", 0),
        "bot_selling_compose": role_counts.get("bot_selling_compose", 0),
        "memory": role_counts.get("memory", 0),
        "judge": role_counts.get("judge", 0),
        "dialogs": int(dialogs),
        "turns": int(turns),
        "avg_calls_per_dialog": round(total / dialogs, 2) if dialogs else None,
    }


def _branch_count_metrics(transcripts: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    template_suffixes = ("_safe_template_applied", "_fallback_applied", "_handoff_applied", "_template_applied")
    template_flags: Counter[str] = Counter()
    p0_flags: Counter[str] = Counter()
    contract_controlled = 0
    turns_count = 0
    for dialog in transcripts:
        for turn in dialog.get("turns") or []:
            turns_count += 1
            flags = [str(flag) for flag in (turn.get("bot_safety_flags") or []) if str(flag).strip()]
            for flag in flags:
                if flag.endswith(template_suffixes):
                    template_flags[flag] += 1
                if "zero_collect" in flag or "high_risk" in flag or "p0" in flag or "manager_only" in flag:
                    p0_flags[flag] += 1
            contract = turn.get("bot_answer_contract") if isinstance(turn.get("bot_answer_contract"), Mapping) else {}
            if contract.get("must_answer_first"):
                contract_controlled += 1
    return {
        "turns": turns_count,
        "template_branch_hits": sum(template_flags.values()),
        "template_branch_flags": dict(template_flags),
        "p0_branch_hits": sum(p0_flags.values()),
        "p0_branch_flags": dict(p0_flags),
        "answer_contract_controlled_turns": contract_controlled,
    }


def judge_fact_audit_summary(transcripts: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    levels: Counter[str] = Counter()
    claim_types: Counter[str] = Counter()
    wrong_scope: list[dict[str, Any]] = []
    risky: list[dict[str, Any]] = []
    for dialog in transcripts:
        for turn in dialog.get("turns") or []:
            audit = turn.get("judge_fact_audit") if isinstance(turn.get("judge_fact_audit"), Mapping) else {}
            for item in audit.get("items") or []:
                if not isinstance(item, Mapping):
                    continue
                level = str(item.get("level") or "").strip()
                if level:
                    levels[level] += 1
                claim_type = str(item.get("claim_type") or "").strip()
                if claim_type:
                    claim_types[claim_type] += 1
                if level == "wrong_scope":
                    wrong_scope.append(
                        {
                            "dialog_id": dialog.get("dialog_id"),
                            "turn": turn.get("turn"),
                            "claim_type": claim_type,
                            "reason": item.get("reason") or "",
                            "client_message": turn.get("client_message"),
                            "bot_text": turn.get("bot_text"),
                            "matched_fact_keys": item.get("matched_fact_keys") or [],
                        }
                    )
                if level in {"no_match", "other_brand_match"}:
                    risky.append(
                        {
                            "dialog_id": dialog.get("dialog_id"),
                            "turn": turn.get("turn"),
                            "claim_type": claim_type,
                            "claim_text": item.get("claim_text") or "",
                            "level": level,
                            "client_message": turn.get("client_message"),
                        }
                    )
    return {
        "version": JUDGE_FACT_AUDIT_VERSION,
        "counts_by_level": dict(levels),
        "claim_types": dict(claim_types),
        "wrong_scope_count": len(wrong_scope),
        "risky_claim_count": len(risky),
        "wrong_scope_examples": wrong_scope[:40],
        "risky_claim_examples": risky[:40],
    }


def _over_handoff_metrics(transcripts: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    total_turns = 0
    handoff_turns: list[dict[str, Any]] = []
    false_handoff: list[dict[str, Any]] = []
    levels: Counter[str] = Counter()
    for dialog in transcripts:
        for turn in dialog.get("turns") or []:
            total_turns += 1
            if not _is_over_handoff_turn(turn):
                continue
            level = _handoff_fact_level(turn)
            levels[level] += 1
            item = {
                "dialog_id": dialog.get("dialog_id"),
                "brand": dialog.get("brand"),
                "turn": turn.get("turn"),
                "route": turn.get("bot_route"),
                "fact_level": level,
                "client_message": turn.get("client_message"),
                "fallback_reason": _turn_primary_fallback_reason(turn),
                "retrieved_fact_keys": list(((turn.get("bot_dialogue_contract_pipeline") or {}).get("retrieved_facts") or {}).keys())[:8],
                "missing_fact_keys": list((turn.get("bot_dialogue_contract_pipeline") or {}).get("missing_fact_keys") or [])[:8],
            }
            handoff_turns.append(item)
            if level == "retrieved_match":
                false_handoff.append(item)
    return {
        "turns": total_turns,
        "handoff_turns": len(handoff_turns),
        "over_handoff_turn_rate": round(len(handoff_turns) / total_turns, 3) if total_turns else None,
        "levels": dict(levels),
        "false_handoff_count": len(false_handoff),
        "false_handoff": false_handoff,
        "candidates": handoff_turns[:80],
    }


def _handoff_trace_enabled() -> bool:
    return str(os.getenv(HANDOFF_TRACE_ENV) or "").strip().casefold() in {"1", "true", "yes", "да", "on"}


def _handoff_trace_summary(transcripts: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    traces: list[Mapping[str, Any]] = []
    for dialog in transcripts:
        for turn in dialog.get("turns") or []:
            if not isinstance(turn, Mapping):
                continue
            trace = turn.get("handoff_trace")
            if isinstance(trace, Mapping) and trace:
                traces.append(trace)
    return {
        "count": len(traces),
        "by_layer": dict(Counter(str(item.get("layer") or "") for item in traces)),
        "by_guard": dict(Counter(str(item.get("guard") or "") for item in traces)),
        "by_fallback_reason": dict(Counter(str(item.get("fallback_reason") or "") for item in traces)),
        "by_provider_error": dict(Counter(str(item.get("provider_error") or "") for item in traces if item.get("provider_error"))),
        "by_gate_finding": dict(
            Counter(str(code) for item in traces for code in (item.get("gate_findings") or []) if str(code).strip())
        ),
        "examples": [dict(item) for item in traces[:20]],
    }


def _handoff_trace_for_turn(turn: Mapping[str, Any]) -> Mapping[str, Any]:
    if not _is_over_handoff_turn(turn):
        return {}
    pipeline = turn.get("bot_dialogue_contract_pipeline") if isinstance(turn.get("bot_dialogue_contract_pipeline"), Mapping) else {}
    contract = pipeline.get("contract") if isinstance(pipeline.get("contract"), Mapping) else {}
    rules_engine = pipeline.get("rules_engine") if isinstance(pipeline.get("rules_engine"), Mapping) else {}
    flags = [str(flag) for flag in (turn.get("bot_safety_flags") or []) if str(flag).strip()]
    fallback_reason = str(pipeline.get("fallback_reason") or "")
    provider_error = str(turn.get("bot_provider_error") or "").strip()
    gate = turn.get("bot_authoritative_output_gate") if isinstance(turn.get("bot_authoritative_output_gate"), Mapping) else {}
    gate_findings = _authoritative_gate_finding_codes(gate)
    layer, guard = _handoff_trace_layer_guard(
        route=str(turn.get("bot_route") or ""),
        fallback_reason=fallback_reason,
        provider_error=provider_error,
        gate_findings=gate_findings,
        flags=flags,
        pipeline=pipeline,
        rules_engine=rules_engine,
        contract=contract,
    )
    reason = _handoff_trace_reason(
        fallback_reason=fallback_reason,
        provider_error=provider_error,
        gate_findings=gate_findings,
        flags=flags,
        pipeline=pipeline,
        fact_level=_handoff_fact_level(turn),
    )
    return {
        "layer": layer,
        "guard": guard,
        "fallback_reason": fallback_reason,
        "provider_error": provider_error,
        "gate_findings": list(gate_findings),
        "reason": reason,
        "route": str(turn.get("bot_route") or ""),
        "fact_level": _handoff_fact_level(turn),
    }


def _handoff_trace_layer_guard(
    *,
    route: str,
    fallback_reason: str,
    provider_error: str,
    gate_findings: Sequence[str],
    flags: Sequence[str],
    pipeline: Mapping[str, Any],
    rules_engine: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> tuple[str, str]:
    joined_flags = " ".join(flags).casefold()
    if contract.get("is_p0") or re.search(r"high_risk|p0|zero_collect|refund", joined_flags, re.I):
        return "safety", "p0_or_high_risk"
    if fallback_reason:
        if fallback_reason in {"hard_verification_failed", "semantic_check_unavailable", "estimate_guard_failed"}:
            return "dialogue_contract_pipeline", fallback_reason
        if fallback_reason.startswith("estimate_"):
            return "dialogue_contract_pipeline", "estimate"
        if fallback_reason.startswith("empty_facts") or fallback_reason in {"contract_manager_only", "no_draft_fn", "draft_error"}:
            return "dialogue_contract_pipeline", fallback_reason
        return "dialogue_contract_pipeline", fallback_reason
    if gate_findings:
        return "authoritative_output_gate", ",".join(gate_findings[:5])
    if provider_error:
        return "provider_runtime", provider_error[:80]
    if re.search(r"cross_brand|brand_separation", joined_flags, re.I):
        return "guard_chain", "brand_separation"
    if re.search(r"guarantee|unsupported_promise|promocode|placeholder|identity", joined_flags, re.I):
        return "guard_chain", "output_safety"
    if rules_engine:
        return "rules_engine", str(rules_engine.get("applied") or rules_engine.get("subvariant") or "domain_rule")
    findings = pipeline.get("findings") if isinstance(pipeline.get("findings"), Sequence) else ()
    if findings:
        return "dialogue_contract_pipeline", "output_verifier"
    if route == "manager_only":
        return "route_policy", "manager_only"
    return "handoff_text", "handoff_phrase"


def _handoff_trace_reason(
    *,
    fallback_reason: str,
    provider_error: str,
    gate_findings: Sequence[str],
    flags: Sequence[str],
    pipeline: Mapping[str, Any],
    fact_level: str,
) -> str:
    if fallback_reason:
        return fallback_reason
    if gate_findings:
        return "authoritative_output_gate:" + ",".join(gate_findings[:5])
    if provider_error:
        return provider_error
    findings = pipeline.get("findings") if isinstance(pipeline.get("findings"), Sequence) else ()
    finding_codes = [
        str(item.get("code") or "")
        for item in findings
        if isinstance(item, Mapping) and str(item.get("code") or "").strip()
    ]
    if finding_codes:
        return "findings:" + ",".join(finding_codes[:5])
    if flags:
        return "flags:" + ",".join(flags[:5])
    if fact_level:
        return "fact_level:" + fact_level
    return "handoff_route_or_text"


def _is_over_handoff_turn(turn: Mapping[str, Any]) -> bool:
    if _turn_is_real_p0(turn):
        return False
    route = str(turn.get("bot_route") or "")
    text = str(turn.get("bot_text") or "").casefold()
    if route in {"draft_for_manager", "manager_only"}:
        return True
    return bool(
        re.search(
            r"^\s*(передам|(?:менеджер|сотрудник)\s+(?:уточнит|подтвердит|сверит|ответит|свяжется|напишет))",
            text,
            re.I,
        )
    )


def _turn_is_real_p0(turn: Mapping[str, Any]) -> bool:
    pipeline = turn.get("bot_dialogue_contract_pipeline") if isinstance(turn.get("bot_dialogue_contract_pipeline"), Mapping) else {}
    contract = pipeline.get("contract") if isinstance(pipeline.get("contract"), Mapping) else {}
    fallback = str(pipeline.get("fallback_reason") or "")
    flags = " ".join(str(flag) for flag in (turn.get("bot_safety_flags") or []))
    return bool(contract.get("is_p0")) or fallback.startswith("p0") or bool(re.search(r"high_risk|p0|manager_only_p0", flags, re.I))


def _handoff_fact_level(turn: Mapping[str, Any]) -> str:
    pipeline = turn.get("bot_dialogue_contract_pipeline") if isinstance(turn.get("bot_dialogue_contract_pipeline"), Mapping) else {}
    contract = pipeline.get("contract") if isinstance(pipeline.get("contract"), Mapping) else {}
    retrieved = pipeline.get("retrieved_facts") if isinstance(pipeline.get("retrieved_facts"), Mapping) else {}
    missing = set(str(item) for item in (pipeline.get("missing_fact_keys") or []))
    if _turn_has_retrieved_match_for_contract(contract, retrieved, missing):
        return "retrieved_match"
    audit = turn.get("number_audit") if isinstance(turn.get("number_audit"), Mapping) else {}
    audit_levels = set(str(item.get("level") or "") for item in (audit.get("items") or []) if isinstance(item, Mapping))
    if "same_brand_global_match" in audit_levels:
        return "same_brand_global_match"
    if retrieved:
        return "wrong_scope"
    return "no_match"


def _turn_has_retrieved_match_for_contract(
    contract: Mapping[str, Any],
    retrieved_facts: Mapping[str, Any],
    missing: set[str],
) -> bool:
    if not contract or not retrieved_facts:
        return False
    retrieved_keys = tuple(str(key) for key in retrieved_facts.keys())
    subquestions = contract.get("subquestions") if isinstance(contract.get("subquestions"), Sequence) else []
    if not subquestions:
        subquestions = (contract,)
    for subquestion in subquestions:
        if not isinstance(subquestion, Mapping):
            continue
        required = tuple(str(key) for key in (subquestion.get("needed_fact_keys") or []) if str(key))
        if not required:
            continue
        if any(key in missing for key in required):
            continue
        if all(any(_fact_key_matches_required(key, retrieved_key) for retrieved_key in retrieved_keys) for key in required) and _summary_scope_exact(
            contract,
            subquestion,
            required,
            retrieved_facts,
        ):
            return True
    return False


def _summary_scope_exact(
    contract: Mapping[str, Any],
    subquestion: Mapping[str, Any],
    required: Sequence[str],
    retrieved_facts: Mapping[str, Any],
) -> bool:
    text = " ".join(
        str(part or "")
        for part in (
            contract.get("current_question"),
            contract.get("existence_target"),
            subquestion.get("text"),
            subquestion.get("existence_target"),
        )
    ).casefold().replace("ё", "е")
    fact_text = " ".join(f"{key} {value}" for key, value in retrieved_facts.items()).casefold().replace("ё", "е")
    if re.search(r"(прям\w*\s+перевод|перевод\w*\s+на\s+счет|перевод\w*\s+на\s+сч[её]т|по\s+счету|по\s+сч[её]ту|напрямую\s+(?:вам|центру)|без\s+банка|вам\s+платить)", text, re.I):
        return bool(re.search(r"(прям\w*\s+перевод|перевод\w*\s+на\s+счет|перевод\w*\s+на\s+сч[её]т|по\s+счету|по\s+сч[её]ту|реквизит|квитанц|qr-?код|qr\s)", fact_text, re.I))
    if re.search(r"помесячн\w*.*сумм|сумм\w*\s+в\s+месяц|сколько\s+.*(?:в|за)\s+месяц|месячн\w*\s+сумм", text, re.I):
        return bool(
            re.search(
                r"сумм\w*\s+в\s+месяц|ежемесячн\w*\s+сумм|помесячн\w*\s+сумм|руб\w*\s+в\s+месяц|₽\s*/\s*мес",
                fact_text,
                re.I,
            )
        )
    if re.search(r"выходн|суббот|воскрес|будн|по\s+каким\s+дням|дни\s+занят", text, re.I):
        if re.search(r"objection|возраж", fact_text, re.I):
            return False
        return bool(re.search(r"выходн|суббот|воскрес|будн", fact_text, re.I))
    if re.search(r"возврат|верн[её]т|вернут|передума", text, re.I):
        return bool(re.search(r"возврат|неистраченн", fact_text, re.I))
    return True


def _fact_key_matches_required(required: str, fact_key: str) -> bool:
    if required == fact_key:
        return True
    try:
        return bool(key_matches(required, fact_key))
    except Exception:
        required_norm = re.sub(r"[^a-zа-яё0-9]+", "", required.casefold())
        fact_norm = re.sub(r"[^a-zа-яё0-9]+", "", fact_key.casefold())
        return bool(required_norm and (required_norm in fact_norm or fact_norm in required_norm))


def _send_unedited_proxy(
    transcripts: Sequence[Mapping[str, Any]],
    judge_results: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    judge_by_id = {str(item.get("dialog_id") or ""): item for item in judge_results}
    candidate_turns = 0
    unedited_turns = 0
    for dialog in transcripts:
        judge = judge_by_id.get(str(dialog.get("dialog_id") or "")) or {}
        if str(judge.get("verdict") or "").upper() == "FAIL" or not judge.get("hard_gates_passed", True):
            continue
        for turn in dialog.get("turns") or []:
            if str(turn.get("bot_route") or "") not in {"bot_answer_self", "bot_answer_self_for_pilot"}:
                continue
            candidate_turns += 1
            if not turn.get("bot_answer_quality_rewritten") and not (turn.get("bot_answer_quality_findings") or []):
                unedited_turns += 1
    return {
        "candidate_autonomous_turns": candidate_turns,
        "unedited_autonomous_turns": unedited_turns,
        "unedited_rate": round(unedited_turns / candidate_turns, 3) if candidate_turns else None,
        "unedited_rate_ci": wilson_interval(unedited_turns, candidate_turns) if candidate_turns else None,
    }


def build_metric_intervals(
    *,
    dialogs: int,
    pass_count: int,
    hard_gate_pass_count: int,
    tone_scores: Sequence[int],
    send_unedited: Mapping[str, Any],
) -> Mapping[str, Any]:
    send_rate = send_unedited.get("unedited_rate")
    intervals: dict[str, Any] = {
        "dialog_pass_rate": {
            "level": "dialog",
            "numerator": pass_count,
            "denominator": dialogs,
            "value": round(pass_count / dialogs, 3) if dialogs else None,
            "ci": wilson_interval(pass_count, dialogs) if dialogs else None,
            "target": METRIC_TARGETS["pass_rate"],
        },
        "hard_gate_pass_rate": {
            "level": "dialog",
            "numerator": hard_gate_pass_count,
            "denominator": dialogs,
            "value": round(hard_gate_pass_count / dialogs, 3) if dialogs else None,
            "ci": wilson_interval(hard_gate_pass_count, dialogs) if dialogs else None,
            "target": METRIC_TARGETS["hard_gate_pass_rate"],
        },
        "send_unedited_rate": {
            "level": "turn",
            "numerator": send_unedited.get("unedited_autonomous_turns"),
            "denominator": send_unedited.get("candidate_autonomous_turns"),
            "value": send_rate,
            "ci": send_unedited.get("unedited_rate_ci"),
            "target": METRIC_TARGETS["send_unedited_rate"],
        },
        "human_tone_score": tone_stats(tone_scores),
    }
    intervals["human_tone_score"]["target"] = METRIC_TARGETS["avg_human_tone_score"]
    reasons: list[str] = []
    for key in ("dialog_pass_rate", "hard_gate_pass_rate", "send_unedited_rate"):
        metric = intervals[key]
        ci = metric.get("ci")
        value = metric.get("value")
        target = metric.get("target")
        if value is None or not isinstance(ci, Mapping) or target is None:
            continue
        if ci.get("low") <= target <= ci.get("high"):
            reasons.append(f"{key}_ci_crosses_target")
    hard_metric = intervals["hard_gate_pass_rate"]
    if hard_metric.get("value") is not None and hard_metric.get("value") < 1:
        reasons.append("hard_gate_failure_observed")
    tone = intervals["human_tone_score"]
    tone_mean = tone.get("mean")
    tone_target = tone.get("target")
    if tone_mean is not None and tone_target is not None:
        low = tone_mean - float(tone.get("stderr") or 0.0)
        high = tone_mean + float(tone.get("stderr") or 0.0)
        if low <= tone_target <= high:
            reasons.append("human_tone_stderr_crosses_target")
    return {
        "metrics_intervals": intervals,
        "needs_second_run": bool(reasons),
        "needs_second_run_reasons": reasons,
    }


def wilson_interval(successes: int, total: int, *, z: float = 1.96) -> Mapping[str, float] | None:
    if total <= 0:
        return None
    p = successes / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denom
    return {"low": round(max(0.0, center - margin), 3), "high": round(min(1.0, center + margin), 3)}


def tone_stats(scores: Sequence[int]) -> Mapping[str, Any]:
    values = [float(score) for score in scores]
    if not values:
        return {"level": "dialog", "n": 0, "mean": None, "stddev": None, "stderr": None}
    mean = sum(values) / len(values)
    if len(values) > 1:
        variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
        stddev = math.sqrt(variance)
        stderr = stddev / math.sqrt(len(values))
    else:
        stddev = 0.0
        stderr = 0.0
    return {
        "level": "dialog",
        "n": len(values),
        "mean": round(mean, 1),
        "stddev": round(stddev, 2),
        "stderr": round(stderr, 2),
    }


def compact_confirmed_facts(context: Mapping[str, Any], *, limit: int = 12, max_chars: int = 220) -> list[str]:
    facts = context.get("confirmed_facts") if isinstance(context, Mapping) else {}
    if not isinstance(facts, Mapping):
        return []
    result: list[str] = []
    for key, value in facts.items():
        text = str(value or "").strip()
        if not text:
            continue
        if len(text) > max_chars:
            text = text[: max_chars - 1].rstrip() + "…"
        result.append(f"{key}: {text}")
        if len(result) >= limit:
            break
    return result


def _dialogue_contract_metadata_from_result(result: Any) -> Mapping[str, Any]:
    metadata = getattr(result, "metadata", None)
    if not isinstance(metadata, Mapping):
        return {}
    pipeline = metadata.get("dialogue_contract_pipeline")
    return dict(pipeline) if isinstance(pipeline, Mapping) else {}


_BOT_SELF_ROUTES = {"bot_answer_self", "bot_answer_self_for_pilot"}
_VISIBLE_ADVICE_REASON_CODES = {"estimate_individual_child_advice", "estimate_general_advice_risk"}


def _manager_deferral_metadata_from_result(
    result: Any,
    *,
    dialogue_contract_metadata: Mapping[str, Any],
    authoritative_gate_metadata: Mapping[str, Any],
) -> Mapping[str, Any]:
    route = str(getattr(result, "route", "") or "")
    if route in _BOT_SELF_ROUTES:
        return {"is_manager_deferral": False, "reason_class": "", "reason_evidence": {}}

    pipeline_deferral = dialogue_contract_metadata.get("is_manager_deferral")
    pipeline_reason_class = str(dialogue_contract_metadata.get("reason_class") or "").strip()
    pipeline_evidence = (
        dict(dialogue_contract_metadata.get("reason_evidence") or {})
        if isinstance(dialogue_contract_metadata.get("reason_evidence"), Mapping)
        else {}
    )
    if pipeline_deferral is not None and pipeline_reason_class:
        return {
            "is_manager_deferral": bool(pipeline_deferral),
            "reason_class": pipeline_reason_class,
            "reason_evidence": pipeline_evidence,
        }

    provider_error = str(getattr(result, "error", "") or "").strip()
    gate_codes = _authoritative_gate_finding_codes(authoritative_gate_metadata)
    reason_class = _reason_class_from_runtime_channels(
        fallback_reason=str(dialogue_contract_metadata.get("fallback_reason") or "").strip(),
        provider_error=provider_error,
        gate_codes=gate_codes,
        safety_flags=tuple(str(flag) for flag in (getattr(result, "safety_flags", ()) or ())),
    )
    evidence: dict[str, Any] = {}
    if dialogue_contract_metadata.get("fallback_reason"):
        evidence["fallback_reason"] = str(dialogue_contract_metadata.get("fallback_reason") or "")
    if gate_codes:
        evidence["gate_findings"] = list(gate_codes)
    if provider_error:
        evidence["provider_error"] = provider_error
    return {"is_manager_deferral": True, "reason_class": reason_class, "reason_evidence": evidence}


def _reason_class_from_runtime_channels(
    *,
    fallback_reason: str,
    provider_error: str,
    gate_codes: Sequence[str],
    safety_flags: Sequence[str],
) -> str:
    if provider_error:
        return "provider_runtime"
    for code in gate_codes:
        if code in _VISIBLE_ADVICE_REASON_CODES:
            return code
    if gate_codes:
        return "output_safety"
    reason = str(fallback_reason or "").strip().casefold()
    flags = " ".join(str(item or "") for item in safety_flags).casefold()
    if re.search(r"p0|zero_collect|refund_claim|complaint", flags, re.I) or reason.startswith("p0"):
        return "p0_deferral"
    if "refund" in reason:
        return "refund"
    if "high_risk" in reason:
        return "high_risk"
    if reason in {"low_confidence", "no_fact_or_unverified", "policy_permission", "payment", "terminal"}:
        return reason
    if reason in {"semantic_check_unavailable", "draft_error", "no_draft_fn"}:
        return "provider_runtime"
    if reason in {"hard_verification_failed", "authoritative_output_gate_blocked"}:
        return "output_safety"
    if reason in {"empty_facts_no_fabrication", "estimate_guard_failed"}:
        return "no_fact_or_unverified"
    if reason:
        return reason
    return "policy_permission"


def _authoritative_output_gate_metadata_from_result(result: Any) -> Mapping[str, Any]:
    metadata = getattr(result, "metadata", None)
    if not isinstance(metadata, Mapping):
        return {}
    gate = metadata.get("authoritative_output_gate")
    if not isinstance(gate, Mapping):
        return {}
    findings = gate.get("findings")
    compact_findings: list[Mapping[str, str]] = []
    if isinstance(findings, Sequence) and not isinstance(findings, (str, bytes, bytearray)):
        for item in findings[:8]:
            if not isinstance(item, Mapping):
                continue
            code = str(item.get("code") or "").strip()
            if not code:
                continue
            compact_findings.append(
                {
                    "code": code,
                    "policy": str(item.get("policy") or "").strip(),
                    "source": str(item.get("source") or "").strip(),
                }
            )
    return {
        "checked": bool(gate.get("checked")),
        "action": str(gate.get("action") or "").strip(),
        "route_before": str(gate.get("route_before") or "").strip(),
        "route_after": str(gate.get("route_after") or "").strip(),
        "findings": compact_findings,
    }


def _authoritative_gate_finding_codes(gate: Mapping[str, Any]) -> tuple[str, ...]:
    findings = gate.get("findings") if isinstance(gate, Mapping) else ()
    if not isinstance(findings, Sequence) or isinstance(findings, (str, bytes, bytearray)):
        return ()
    codes: list[str] = []
    for item in findings:
        if not isinstance(item, Mapping):
            continue
        code = str(item.get("code") or "").strip()
        if code:
            codes.append(code)
    return tuple(dict.fromkeys(codes))


def facts_for_judge(
    context: Mapping[str, Any],
    *,
    dialogue_contract_metadata: Mapping[str, Any] | None = None,
    limit: int = 20,
    max_chars: int = 320,
) -> list[str]:
    """Facts visible to judge should match what the bot actually retrieved.

    For v2 turns, `retrieved_facts` is the cleanest source of truth. We append the
    legacy context facts after it for continuity, but do not replace v2 facts with
    the older selector output.
    """
    result: list[str] = []
    seen: set[str] = set()
    pipeline = dialogue_contract_metadata if isinstance(dialogue_contract_metadata, Mapping) else {}
    retrieved = pipeline.get("retrieved_facts") if isinstance(pipeline.get("retrieved_facts"), Mapping) else {}
    for key, value in retrieved.items():
        text = _compact_fact_for_judge(key, value, max_chars=max_chars)
        if text and text not in seen:
            result.append(text)
            seen.add(text)
    for item in compact_confirmed_facts(context, limit=limit, max_chars=max_chars):
        if item not in seen:
            result.append(item)
            seen.add(item)
        if len(result) >= limit:
            break
    return result


def _compact_fact_for_judge(key: object, value: object, *, max_chars: int) -> str:
    clean_key = str(key or "").strip()
    text = str(value or "").strip()
    if not clean_key or not text:
        return ""
    if len(text) > max_chars:
        text = text[: max_chars - 1].rstrip() + "…"
    return f"{clean_key}: {text}"


def _compact_dialogue_contract_for_judge(value: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    keep = {
        "retrieved_fact_keys": value.get("retrieved_fact_keys") or [],
        "missing_fact_keys": value.get("missing_fact_keys") or [],
        "fallback_reason": value.get("fallback_reason") or "",
        "warmed": bool(value.get("warmed")),
        "repaired": bool(value.get("repaired")),
    }
    return keep


def audit_number_claims(
    text: str,
    *,
    client_message: str,
    active_brand: str,
    retrieved_facts: Mapping[str, Any] | None,
    snapshot_path: Path,
) -> Mapping[str, Any]:
    claims = extract_number_claims(text)
    client_values = {claim["normalized"] for claim in extract_number_claims(client_message)}
    retrieved = {
        str(key): str(value)
        for key, value in (retrieved_facts or {}).items()
        if str(key).strip() and str(value).strip()
    }
    snapshot_index = snapshot_number_index(snapshot_path)
    brand = str(active_brand or "").casefold()
    items: list[dict[str, Any]] = []
    for claim in claims:
        normalized = str(claim["normalized"])
        retrieved_matches = [
            key
            for key, fact_text in retrieved.items()
            if claim_matches_text(claim, fact_text)
        ]
        same_brand_matches = sorted(snapshot_index.get(brand, {}).get(normalized, set()))
        other_brand_matches = sorted(
            key
            for item_brand, values in snapshot_index.items()
            if item_brand != brand
            for key in values.get(normalized, set())
        )
        if str(claim.get("kind") or "") == "weekly_frequency" and int(float(normalized)) > 7:
            level = "kb_integrity_issue"
        elif normalized in client_values:
            level = "client_echo"
        elif retrieved_matches:
            level = "retrieved_match"
        elif same_brand_matches:
            level = "same_brand_global_match"
        elif other_brand_matches:
            level = "other_brand_match"
        else:
            level = "no_match"
        items.append(
            {
                "claim_text": claim["text"],
                "kind": claim["kind"],
                "normalized": normalized,
                "level": level,
                "matched_fact_keys": retrieved_matches or same_brand_matches[:5] or other_brand_matches[:5],
                "matched_brand": brand if retrieved_matches or same_brand_matches else ("other" if other_brand_matches else ""),
            }
        )
    counts = Counter(str(item["level"]) for item in items)
    return {
        "items": items,
        "counts_by_level": dict(counts),
        "worst_level": worst_number_audit_level(counts.keys()),
        "has_risky_number": any(level in {"kb_integrity_issue", "no_match", "other_brand_match"} for level in counts),
    }


_MONEY_AUDIT_RE = re.compile(r"(?<!\d)(\d[\d \u00a0]{2,})(?:\s*(?:₽|руб\.?|рубл(?:ей|я|ь)?))", re.I)
_PERCENT_AUDIT_RE = re.compile(r"(?<!\d)(\d{1,3}(?:[.,]\d+)?)\s*%")
_WEEKLY_AUDIT_RE = re.compile(r"(?<!\d)(\d{1,4})\s+раз(?:а)?\s+в\s+недел", re.I)
_DATE_AUDIT_RE = re.compile(r"(?<!\d)(\d{1,2})[. ](?:\d{1,2}|январ|феврал|март|апрел|ма[йя]|июн|июл|август|сентябр|октябр|ноябр|декабр)", re.I)
_PLAIN_AUDIT_RE = re.compile(r"(?<![\w@/.-])(\d[\d \u00a0]{1,})(?![\w@/.-])")


def extract_number_claims(text: str) -> list[Mapping[str, str]]:
    source = str(text or "")
    claims: list[dict[str, str]] = []
    spans: list[tuple[int, int]] = []
    ignored_spans = list(ignored_number_spans(source))
    for kind, regex in (
        ("money", _MONEY_AUDIT_RE),
        ("percent", _PERCENT_AUDIT_RE),
        ("weekly_frequency", _WEEKLY_AUDIT_RE),
        ("date", _DATE_AUDIT_RE),
    ):
        for match in regex.finditer(source):
            if any(start <= match.start() < end for start, end in ignored_spans):
                continue
            normalized = normalize_date_claim(match.group(0)) if kind == "date" else normalize_audit_number(match.group(1))
            if normalized:
                claims.append({"kind": kind, "text": match.group(0), "normalized": normalized})
                spans.append(match.span())
    for match in _PLAIN_AUDIT_RE.finditer(source):
        if any(start <= match.start() < end for start, end in spans + ignored_spans):
            continue
        normalized = normalize_audit_number(match.group(1))
        if not normalized:
            continue
        value = int(normalized)
        if value in {2026, 2027}:
            continue
        claims.append({"kind": "plain_number", "text": match.group(0), "normalized": normalized})
    return claims


def ignored_number_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for regex in (
        re.compile(r"https?://\S+|www\.\S+|\S+@\S+"),
        re.compile(r"(?:\+7|8)[\s()_-]*\d[\d\s()_-]{7,}\d"),
        re.compile(r"\b20\d{2}\s*/\s*\d{2}\b"),
        re.compile(r"\b\d{1,2}(?:\s*(?:,|и|/|[–-])\s*\d{1,2})+\s+класс\w*", re.I),
        re.compile(r"\b\d{1,2}\s+класс\w*", re.I),
        re.compile(r"\b(?:\d{1,2}[:.]\d{2})\s*[–-]\s*(?:\d{1,2}[:.]\d{2})\b"),
        re.compile(r"\b(?:Л\d+|№\s*\d+|\d+Л\d+)\b", re.I),
    ):
        spans.extend(match.span() for match in regex.finditer(str(text or "")))
    return spans


def normalize_audit_number(value: object) -> str:
    raw = str(value or "").replace(" ", "").replace("\u00a0", "").replace(",", ".")
    try:
        number = float(raw)
    except ValueError:
        return ""
    return str(int(number)) if number == int(number) else str(number)


_MONTH_NUMBER_BY_TEXT = {
    "январ": "01",
    "феврал": "02",
    "март": "03",
    "апрел": "04",
    "ма": "05",
    "июн": "06",
    "июл": "07",
    "август": "08",
    "сентябр": "09",
    "октябр": "10",
    "ноябр": "11",
    "декабр": "12",
}


def normalize_date_claim(value: object) -> str:
    text = str(value or "").casefold().replace("ё", "е")
    match = re.search(
        r"(?<!\d)(\d{1,2})[. ](0?\d{1,2}|январ\w*|феврал\w*|март\w*|апрел\w*|ма[йя]|июн\w*|июл\w*|август\w*|сентябр\w*|октябр\w*|ноябр\w*|декабр\w*)(?:[. ](20\d{2}))?",
        text,
        re.I,
    )
    if not match:
        return ""
    day = int(match.group(1))
    if day < 1 or day > 31:
        return ""
    month_raw = match.group(2)
    month = ""
    if month_raw.isdigit():
        month_number = int(month_raw)
        if 1 <= month_number <= 12:
            month = f"{month_number:02d}"
    else:
        for prefix, number in _MONTH_NUMBER_BY_TEXT.items():
            if month_raw.startswith(prefix):
                month = number
                break
    if not month:
        return ""
    year = match.group(3) or ""
    return f"date:{day:02d}.{month}" + (f".{year}" if year else "")


def claim_matches_text(claim: Mapping[str, Any], text: str) -> bool:
    normalized = str(claim.get("normalized") or "")
    if not normalized:
        return False
    text_numbers = {str(item["normalized"]) for item in extract_number_claims(text)}
    if normalized in text_numbers:
        return True
    return normalized in re.sub(r"\D+", " ", str(text or "")).split()


@lru_cache(maxsize=8)
def snapshot_number_index(snapshot_path: Path) -> Mapping[str, Mapping[str, frozenset[str]]]:
    path = Path(snapshot_path)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    facts = payload.get("facts") if isinstance(payload, Mapping) else []
    index: dict[str, dict[str, set[str]]] = {}
    if not isinstance(facts, Sequence) or isinstance(facts, (str, bytes, bytearray)):
        return {}
    for fact in facts:
        if not isinstance(fact, Mapping):
            continue
        if not fact.get("allowed_for_client_answer"):
            continue
        brand = str(fact.get("brand") or "").casefold()
        key = str(fact.get("fact_key") or fact.get("fact_id") or "")
        text = " ".join(str(fact.get(field) or "") for field in ("client_safe_text", "fact_text", "manager_check_text", "structured_value"))
        for claim in extract_number_claims(text):
            index.setdefault(brand, {}).setdefault(str(claim["normalized"]), set()).add(key)
    return {brand: {number: frozenset(keys) for number, keys in values.items()} for brand, values in index.items()}


def worst_number_audit_level(levels: Any) -> str:
    present = {str(level) for level in levels if str(level)}
    for level in ("kb_integrity_issue", "other_brand_match", "no_match", "same_brand_global_match", "retrieved_match", "client_echo"):
        if level in present:
            return level
    return ""


def _filter_judge_confirmed_facts(items: Sequence[Any]) -> list[str]:
    result: list[str] = []
    for item in items:
        text = str(item or "").strip()
        if not text:
            continue
        if text.casefold().startswith("verified_safe_template:"):
            continue
        result.append(text)
    return result


def compact_knowledge_snippets(context: Mapping[str, Any], *, limit: int = 8, max_chars: int = 220) -> list[str]:
    snippets = context.get("knowledge_snippets") if isinstance(context, Mapping) else []
    if not isinstance(snippets, Sequence) or isinstance(snippets, (str, bytes)):
        return []
    result: list[str] = []
    for value in snippets:
        text = str(value or "").strip()
        if not text:
            continue
        if len(text) > max_chars:
            text = text[: max_chars - 1].rstrip() + "…"
        result.append(text)
        if len(result) >= limit:
            break
    return result


def render_summary_md(summary: Mapping[str, Any]) -> str:
    totals = summary.get("totals") if isinstance(summary.get("totals"), Mapping) else {}
    llm_calls = summary.get("llm_calls") if isinstance(summary.get("llm_calls"), Mapping) else {}
    over_handoff = summary.get("over_handoff") if isinstance(summary.get("over_handoff"), Mapping) else {}
    handoff_trace = summary.get("handoff_trace") if isinstance(summary.get("handoff_trace"), Mapping) else {}
    judge_fact_audit = summary.get("judge_fact_audit") if isinstance(summary.get("judge_fact_audit"), Mapping) else {}
    return "\n".join(
        [
            "# Dynamic Telegram Client Simulation v7",
            "",
            f"- Диалогов: `{totals.get('dialogs')}`",
            f"- Ходов: `{totals.get('turns')}`",
            f"- PASS: `{totals.get('pass')}`",
            f"- PASS_WITH_NOTES: `{totals.get('pass_with_notes')}`",
            f"- FAIL: `{totals.get('fail')}`",
            f"- Hard-gate failures: `{totals.get('hard_gate_failures')}`",
            f"- Средний human tone: `{totals.get('avg_human_tone_score')}`",
            f"- Violated gates: `{summary.get('violated_gates')}`",
            f"- Soft flags: `{summary.get('soft_flags')}`",
            f"- Answer quality: `{summary.get('answer_quality')}`",
            f"- Scenario metadata: `{summary.get('scenario_metadata')}`",
            f"- Branch metrics: `{summary.get('branch_metrics')}`",
            f"- LLM calls: `{llm_calls}`",
            f"- Turn fallback reasons: `{summary.get('turn_fallback_reasons')}`",
            f"- Claude CLI errors: `{summary.get('claude_cli_errors')}`",
            f"- Send-unedited proxy: `{summary.get('send_unedited_proxy')}`",
            f"- Over-handoff: `{over_handoff}`",
            f"- Handoff trace: `{handoff_trace}`",
            f"- Judge fact audit: `{judge_fact_audit}`",
            "",
            "Что смотреть вручную:",
            "",
            "- `human_review_queue.csv` — очередь проверки, самые рискованные диалоги сверху.",
            "- `full_transcripts.md` — полный человекочитаемый разговор по всем персонажам.",
            "- `transcripts_md/*.md` — отдельный файл на каждый диалог.",
            "- `dynamic_dialog_transcripts.jsonl` — полный машинный лог всех ходов.",
            "- `dynamic_turns.csv` — плоская таблица ходов с маршрутами и флагами.",
            "- `dynamic_judge_results.jsonl` — вердикт судьи по каждому диалогу.",
            "",
        ]
    )


def render_full_transcripts_md(transcripts: Sequence[Mapping[str, Any]]) -> str:
    chunks = ["# Dynamic Telegram Client Simulation v7: Full Transcripts", ""]
    for dialog in transcripts:
        chunks.append(render_one_dialog_md(dialog))
        chunks.append("")
        chunks.append("---")
        chunks.append("")
    return "\n".join(chunks).rstrip() + "\n"


def render_one_dialog_md(dialog: Mapping[str, Any]) -> str:
    persona = dialog.get("persona") if isinstance(dialog.get("persona"), Mapping) else {}
    judge = dialog.get("judge_result") if isinstance(dialog.get("judge_result"), Mapping) else {}
    lines = [
        f"## {dialog.get('dialog_id') or 'dialog'}",
        "",
        f"- Бренд: `{dialog.get('brand')}`",
        f"- Персона: {persona.get('persona') or ''}",
        f"- Цель клиента: {persona.get('goal') or ''}",
        f"- Verdict: `{judge.get('verdict') or ''}`",
        f"- Hard gates passed: `{judge.get('hard_gates_passed')}`",
        f"- Violated gates: `{format_list(judge.get('violated_gates') or [])}`",
        f"- Soft flags: `{format_list(judge.get('soft_flags_present') or [])}`",
        f"- First failing turn: `{judge.get('first_failing_turn')}`",
            f"- Human tone score: `{judge.get('human_tone_score_0_100')}`",
            f"- Rationale: {judge.get('rationale') or ''}",
        "",
    ]
    for turn in dialog.get("turns") or []:
        lines.extend(
            [
                f"### Ход {turn.get('turn')}",
                "",
                "**Клиент:**",
                "",
                str(turn.get("client_message") or "").strip(),
                "",
                "**Бот:**",
                "",
                str(turn.get("bot_text") or "").strip(),
                "",
                "**Метаданные хода:**",
                "",
                f"- route: `{turn.get('bot_route')}`",
                f"- topic: `{turn.get('bot_topic_id')}`",
                f"- conversation_intent: `{(turn.get('bot_conversation_intent_plan') or {}).get('primary_intent') if isinstance(turn.get('bot_conversation_intent_plan'), Mapping) else ''}`",
                f"- conversation_topic_switch: `{(turn.get('bot_conversation_intent_plan') or {}).get('topic_switch_decision') if isinstance(turn.get('bot_conversation_intent_plan'), Mapping) else ''}`",
                f"- answer_contract_intent: `{(turn.get('bot_answer_contract') or {}).get('primary_intent') if isinstance(turn.get('bot_answer_contract'), Mapping) else ''}`",
                f"- answer_contract_direct_question: `{(turn.get('bot_answer_contract') or {}).get('direct_question') if isinstance(turn.get('bot_answer_contract'), Mapping) else ''}`",
                f"- answer_contract_p0_required: `{(turn.get('bot_answer_contract') or {}).get('p0_required') if isinstance(turn.get('bot_answer_contract'), Mapping) else ''}`",
                f"- message_type: `{turn.get('bot_message_type')}`",
                f"- risk: `{turn.get('bot_risk_level')}`",
                f"- safety_flags: `{format_list(turn.get('bot_safety_flags') or [])}`",
                f"- fallback_reason: `{turn.get('bot_fallback_reason') or ''}`",
                f"- provider_error: `{turn.get('bot_provider_error') or ''}`",
                f"- claude_cli_error_count: `{turn.get('bot_claude_cli_error_count') or 0}`",
                f"- claude_cli_errors: `{turn.get('bot_claude_cli_errors') or []}`",
                f"- handoff_trace: `{turn.get('handoff_trace') or {}}`",
                f"- manager_checklist: `{format_list(turn.get('bot_manager_checklist') or [])}`",
                f"- missing_facts: `{format_list(turn.get('bot_missing_facts') or [])}`",
                f"- answer_quality_findings: `{format_list(turn.get('bot_answer_quality_findings') or [])}`",
                f"- answer_quality_rewritten: `{turn.get('bot_answer_quality_rewritten')}`",
                f"- context_parity_checked: `{turn.get('context_parity_checked')}`",
                f"- confirmed_facts_for_judge: `{format_list(turn.get('bot_confirmed_facts') or [])}`",
                f"- knowledge_snippets_for_judge: `{format_list(turn.get('bot_knowledge_snippets') or [])}`",
                f"- judge_fact_audit: `{turn.get('judge_fact_audit') or {}}`",
                f"- client_stop: `{turn.get('client_stop')}`",
                "",
            ]
        )
    return "\n".join(lines).rstrip()


def known_dialog_fields_from_client_messages(messages: Sequence[str], *, active_brand: str) -> Mapping[str, str]:
    client_parts: list[str] = []
    for item in messages:
        for raw_line in str(item or "").splitlines():
            line = raw_line.strip()
            lowered = line.casefold()
            if lowered.startswith("ответ:"):
                continue
            if lowered.startswith("клиент:"):
                line = line.split(":", 1)[1].strip()
            if line:
                client_parts.append(line)
    normalized = "\n".join(client_parts).casefold().replace("ё", "е")
    result: dict[str, str] = {"active_brand": active_brand}
    grade = re.search(r"\b(?P<grade>[1-9]|1[01])\s*(?:класс|кл\.?)\b", normalized)
    if grade:
        result["grade"] = grade.group("grade")
    subjects: list[str] = []
    for marker, canonical in (
        ("математ", "математика"),
        ("физик", "физика"),
        ("информат", "информатика"),
        ("программирован", "программирование"),
        ("русск", "русский язык"),
        ("англий", "английский язык"),
        ("хими", "химия"),
        ("биолог", "биология"),
    ):
        if marker in normalized:
            subjects.append(canonical)
    if subjects:
        result["subject"] = ", ".join(dict.fromkeys(subjects))
    if "онлайн" in normalized:
        result["format"] = "онлайн"
    elif "очно" in normalized or "офлайн" in normalized:
        result["format"] = "очно"
    return result


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = [
        "dialog_id",
        "brand",
        "turn",
        "client_message",
        "bot_text",
        "bot_route",
        "bot_topic_id",
        "bot_conversation_intent",
        "bot_conversation_topic_switch",
        "bot_safety_flags",
        "bot_answer_quality_findings",
        "bot_answer_quality_rewritten",
        "context_parity_checked",
        "client_stop",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_human_review_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = [
        "review_priority",
        "dialog_id",
        "brand",
        "verdict",
        "hard_gates_passed",
        "violated_gates",
        "soft_flags_present",
        "human_tone_score_0_100",
        "first_failing_turn",
        "lead_captured",
        "turns",
        "bot_flags",
        "rationale",
        "persona",
        "goal",
        "hard_gate_cause",
        "hard_gate_cause_evidence",
        "number_audit_worst_level",
        "manual_check_hint",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def format_list(items: Sequence[Any]) -> str:
    return " | ".join(str(item) for item in items if str(item).strip())


def safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned[:120] or "dialog"


def extract_json_object(text: str) -> Mapping[str, Any]:
    raw = str(text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.I)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end <= start:
            raise
        payload = json.loads(raw[start : end + 1])
    if not isinstance(payload, Mapping):
        raise RuntimeError("Expected JSON object")
    return payload


def _codex_env() -> dict[str, str]:
    import os

    env = dict(os.environ)
    env.pop("OPENAI_API_KEY", None)
    return env


if __name__ == "__main__":
    raise SystemExit(main())
