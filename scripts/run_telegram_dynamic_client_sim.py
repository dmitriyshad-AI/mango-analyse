#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from mango_mvp.channels.subscription_llm import (
    SubscriptionDraftResult,
    SubscriptionLlmDraftProvider,
    _is_verified_client_safe_template,
    normalize_subscription_draft_payload,
    strip_internal_service_markers,
)
from mango_mvp.channels.telegram_pilot_context_builder import build_telegram_pilot_context_from_snapshot
from mango_mvp.channels.subscription_llm import AUTONOMY_MATRIX_SAFE_TOPIC_IDS


DEFAULT_V7_PATH = Path("/Users/dmitrijfabarisov/Claude Projects/Foton/mega_smoke_tests_v7_dynamic_sim_2026-05-21/v7_dynamic_client_sim_2026-05-21.jsonl")
DEFAULT_SNAPSHOT = Path("product_data/knowledge_base/kb_release_20260520_v6_3_team_answers/kb_release_v3_snapshot.json")
DEFAULT_OUT_DIR = Path("audits/_inbox/telegram_dynamic_client_sim_v7")
SCHEMA_VERSION = "telegram_dynamic_client_sim_v1_2026_05_21"


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


class CodexJsonModel:
    def __init__(self, *, model: str, reasoning_effort: str, timeout_sec: int, codex_bin: str = "codex") -> None:
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.timeout_sec = timeout_sec
        self.codex_bin = codex_bin

    def generate(self, prompt: str) -> Mapping[str, Any]:
        with tempfile.NamedTemporaryFile(prefix="mango_dynamic_sim_", suffix=".json") as out_file:
            output_path = Path(out_file.name)
            cmd = [
                self.codex_bin,
                "exec",
                "--skip-git-repo-check",
                "--ephemeral",
                "--sandbox",
                "read-only",
                "--model",
                self.model,
            ]
            if self.reasoning_effort:
                cmd.extend(["-c", f'model_reasoning_effort="{self.reasoning_effort}"'])
            cmd.extend(["--output-last-message", str(output_path), "-"])
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run dynamic client simulator against Telegram bot draft logic.")
    parser.add_argument("--scenarios", type=Path, default=DEFAULT_V7_PATH)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--brand", choices=("all", "foton", "unpk"), default="all")
    parser.add_argument("--limit", type=int, default=0, help="Limit personas for smoke runs.")
    parser.add_argument("--max-turns", type=int, default=0, help="Override persona max_turns.")
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
    parser.add_argument("--client-mode", choices=("codex", "fake"), default="codex")
    parser.add_argument("--judge-mode", choices=("codex", "fake"), default="codex")
    parser.add_argument("--bot-mode", choices=("codex", "fake"), default="codex")
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--bot-reasoning", default="medium")
    parser.add_argument("--client-reasoning", default="medium")
    parser.add_argument("--judge-reasoning", default="high")
    parser.add_argument("--timeout-sec", type=int, default=180)
    args = parser.parse_args(argv)

    if "stable_runtime" in args.out_dir.resolve(strict=False).parts:
        raise ValueError("Refusing to write dynamic sim outputs under stable_runtime")

    sim_input = load_dynamic_sim_input(args.scenarios)
    personas = [item for item in sim_input.personas if args.brand == "all" or item.get("brand") == args.brand]
    if args.limit > 0:
        personas = personas[: args.limit]

    client_model = FakeClientModel() if args.client_mode == "fake" else CodexJsonModel(
        model=args.model,
        reasoning_effort=args.client_reasoning,
        timeout_sec=args.timeout_sec,
    )
    judge_model = FakeJudgeModel() if args.judge_mode == "fake" else CodexJsonModel(
        model=args.model,
        reasoning_effort=args.judge_reasoning,
        timeout_sec=args.timeout_sec,
    )
    bot_provider: Any = FakeBotProvider() if args.bot_mode == "fake" else SubscriptionLlmDraftProvider(
        model=args.model,
        reasoning_effort=args.bot_reasoning,
        timeout_sec=args.timeout_sec,
        cache_dir=Path(".codex_local/telegram_dynamic_client_sim/llm_cache"),
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    transcripts_path = args.out_dir / "dynamic_dialog_transcripts.jsonl"
    judge_path = args.out_dir / "dynamic_judge_results.jsonl"
    turns_path = args.out_dir / "dynamic_turns.csv"
    review_queue_path = args.out_dir / "human_review_queue.csv"
    full_transcripts_md_path = args.out_dir / "full_transcripts.md"
    per_dialog_dir = args.out_dir / "transcripts_md"
    summary_path = args.out_dir / "dynamic_summary.json"
    summary_md_path = args.out_dir / "dynamic_summary.md"

    if args.transcripts_in is not None:
        transcripts = [
            attach_context_facts_to_dialog(dialog, snapshot_path=args.snapshot)
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
        completed_ids = {
            str(dialog.get("dialog_id") or "")
            for dialog in transcripts
            if str(dialog.get("dialog_id") or "").strip()
        }
        judge_results = [
            dict(dialog.get("judge_result") or {})
            for dialog in transcripts
            if isinstance(dialog.get("judge_result"), Mapping)
        ]
        for persona in personas:
            dialog_id = str(persona.get("dialog_id") or "")
            if dialog_id in completed_ids:
                print(f"skip_completed_dialog={dialog_id}", flush=True)
                continue
            print(f"run_dialog={dialog_id}", flush=True)
            started = time.time()
            dialog = run_one_dialog(
                persona,
                simulator_spec=sim_input.simulator_spec,
                judge_spec=sim_input.judge_spec,
                client_model=client_model,
                judge_model=judge_model,
                bot_provider=bot_provider,
                snapshot_path=args.snapshot,
                max_turns_override=args.max_turns,
            )
            dialog = {**dialog, "elapsed_seconds": round(time.time() - started, 3)}
            transcripts.append(dialog)
            judge_results.append(dict(dialog["judge_result"]))
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
            )
            print(
                f"done_dialog={dialog_id} elapsed={dialog['elapsed_seconds']}s verdict={dialog['judge_result'].get('verdict')}",
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
    summary = build_summary(transcripts, judge_results, scenario_path=scenario_path, snapshot_path=snapshot_path)
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
                    "bot_safety_flags": "|".join(str(flag) for flag in (turn.get("bot_safety_flags") or [])),
                    "client_stop": turn.get("client_stop"),
                }
            )
    return rows


def attach_context_facts_to_dialog(dialog: Mapping[str, Any], *, snapshot_path: Path) -> Mapping[str, Any]:
    persona = dialog.get("persona") if isinstance(dialog.get("persona"), Mapping) else {}
    recent_messages: list[str] = []
    turns: list[Mapping[str, Any]] = []
    for raw_turn in dialog.get("turns") or []:
        turn = dict(raw_turn)
        client_message = str(turn.get("client_message") or "")
        bot_text = str(turn.get("bot_text") or "")
        context = build_bot_prompt_context(
            client_message,
            persona=persona,
            recent_messages=recent_messages,
            snapshot_path=snapshot_path,
        )
        turn["bot_confirmed_facts"] = compact_confirmed_facts(context)
        turn["bot_knowledge_snippets"] = compact_knowledge_snippets(context)
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
    max_turns_override: int = 0,
) -> Mapping[str, Any]:
    dialog_id = str(persona.get("dialog_id") or "")
    brand = str(persona.get("brand") or "unknown")
    max_turns = int(max_turns_override or persona.get("max_turns") or 6)
    turns: list[dict[str, Any]] = []
    recent_messages: list[str] = []

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
        )
        result = bot_provider.build_draft(client_message, context=context)
        bot_text = strip_internal_service_markers(str(result.draft_text or "")).strip()
        confirmed_facts_for_judge = compact_confirmed_facts(context)
        if _is_verified_client_safe_template(bot_text):
            confirmed_facts_for_judge = [
                f"verified_safe_template: {bot_text}",
                *confirmed_facts_for_judge,
            ]
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
            "bot_confirmed_facts": confirmed_facts_for_judge,
            "bot_knowledge_snippets": compact_knowledge_snippets(context),
        }
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
) -> Mapping[str, Any]:
    brand = str(persona.get("brand") or "unknown")
    known_dialog = known_dialog_fields_from_client_messages([*recent_messages, f"Клиент: {client_message}"], active_brand=brand)
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
        recent_messages=tuple(recent_messages[-10:]),
        client_identity={"channel": "dynamic_sim", "channel_thread_id": str(persona.get("dialog_id") or ""), "channel_user_id": "dynamic_sim"},
        customer_summary=f"Динамический тестовый клиент: {persona.get('persona')}. Не раскрывать это клиенту.",
    )
    payload = dict(pilot_context.to_prompt_context())
    payload["active_brand"] = brand
    payload["known_dialog_fields"] = known_dialog
    payload["dynamic_client_sim"] = {
        "enabled": True,
        "dialog_id": persona.get("dialog_id"),
        "do_not_disclose_simulation": True,
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
        "Подтверждённые факты, доступные боту на этом ходу, клиент их НЕ видел как служебный блок: "
        f"{turn.get('bot_confirmed_facts') or []}\n"
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
        "Не ставь fabrication за факт, который прямо есть в подтверждённых фактах или фрагментах базы на этом ходу. "
        "Но если бот приписывает клиенту предмет/цель/потребность, которую клиент не называл, отмечай soft flag assumed_unstated_need; "
        "если такого предмета/продукта ещё и нет в подтверждённых фактах бренда — это уже hard gate fabrication.\n\n"
        "Инструкция судьи:\n"
        f"{json.dumps(judge_spec, ensure_ascii=False, indent=2)}\n\n"
        "Персона:\n"
        f"{json.dumps(persona, ensure_ascii=False, indent=2)}\n\n"
        "Транскрипт:\n"
        f"{transcript}\n"
    )


def normalize_judge_result(payload: Mapping[str, Any], *, dialog_id: str, brand: str) -> Mapping[str, Any]:
    result = dict(payload)
    result["dialog_id"] = dialog_id
    result["brand"] = brand
    result["hard_gates_passed"] = bool(result.get("hard_gates_passed", False))
    if not isinstance(result.get("violated_gates"), list):
        result["violated_gates"] = []
    if not isinstance(result.get("soft_flags_present"), list):
        result["soft_flags_present"] = []
    if not isinstance(result.get("quality_scores"), Mapping):
        result["quality_scores"] = {}
    try:
        result["human_tone_score_0_100"] = int(result.get("human_tone_score_0_100") or 0)
    except (TypeError, ValueError):
        result["human_tone_score_0_100"] = 0
    verdict = str(result.get("verdict") or "").strip().upper()
    if verdict not in {"PASS", "PASS_WITH_NOTES", "FAIL"}:
        verdict = "FAIL" if not result["hard_gates_passed"] else "PASS_WITH_NOTES"
    result["verdict"] = verdict
    return result


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


def build_summary(
    transcripts: Sequence[Mapping[str, Any]],
    judge_results: Sequence[Mapping[str, Any]],
    *,
    scenario_path: Path,
    snapshot_path: Path,
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
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "scenario_path": str(scenario_path),
        "snapshot_path": str(snapshot_path),
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
        "hard_gate_failure_dialogs": [item.get("dialog_id") for item in hard_gate_failures],
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
                f"- message_type: `{turn.get('bot_message_type')}`",
                f"- risk: `{turn.get('bot_risk_level')}`",
                f"- safety_flags: `{format_list(turn.get('bot_safety_flags') or [])}`",
                f"- manager_checklist: `{format_list(turn.get('bot_manager_checklist') or [])}`",
                f"- missing_facts: `{format_list(turn.get('bot_missing_facts') or [])}`",
                f"- confirmed_facts_for_judge: `{format_list(turn.get('bot_confirmed_facts') or [])}`",
                f"- knowledge_snippets_for_judge: `{format_list(turn.get('bot_knowledge_snippets') or [])}`",
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
        ("программ", "программирование"),
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
        "bot_safety_flags",
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
