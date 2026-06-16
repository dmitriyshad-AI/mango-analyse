#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from mango_mvp.channels.dialogue_memory import MEMORY_PROVENANCE_ENV, build_dialogue_memory
from mango_mvp.channels.new_lead_funnel import ANCHORED_BARE_GRADE_ENV
from mango_mvp.channels.subscription_llm import SubscriptionDraftResult
from mango_mvp.channels.subscription_llm_parts.post_layers import apply_question_instead_of_handoff_layer
from mango_mvp.channels.subscription_llm_parts.support import (
    DIRECT_PATH_ENV,
    DIRECT_PATH_PILOT_CONFIG_ENV,
    DIRECT_PATH_PILOT_CONFIG_VERSION,
    QUESTION_INSTEAD_OF_HANDOFF_ENV,
)


P0_RE = re.compile(r"верните|возврат|жалоб|списал|претензи|обман|суд|полици", re.I)
TARGET_SCHEMA_VERSION = "tz123_target_question_set_v1_2026_06_16"


@dataclass(frozen=True)
class TargetCase:
    case_id: str
    brand: str
    client_message: str
    expected_slot: str
    known_slots: Mapping[str, str]
    facts: Mapping[str, str]
    fact_metadata: Mapping[str, Mapping[str, Any]]
    missing_facts: tuple[str, ...] = ()


TARGET_CASES: tuple[TargetCase, ...] = (
    TargetCase(
        case_id="T01_unpk_price_no_grade",
        brand="unpk",
        client_message="Сколько стоит у вас?",
        expected_slot="grade",
        known_slots={},
        facts={
            "unpk.offline.1_4.price": "УНПК: очные курсы 1-4 классы стоят 31 000 ₽ за семестр.",
            "unpk.offline.5_11.price": "УНПК: очные курсы 5-11 классы стоят 52 900 ₽ за семестр.",
        },
        fact_metadata={},
    ),
    TargetCase(
        case_id="T02_foton_online_price_no_grade",
        brand="foton",
        client_message="Сколько стоит онлайн-обучение?",
        expected_slot="grade",
        known_slots={"format": "онлайн"},
        facts={
            "foton.online.3_4.price": "Фотон: онлайн для 3-4 класса стоит 19 000 ₽ за семестр.",
            "foton.online.8_11.price": "Фотон: онлайн для 8-11 класса стоит 47 250 ₽ за семестр.",
        },
        fact_metadata={},
    ),
    TargetCase(
        case_id="T03_unpk_monthly_no_grade",
        brand="unpk",
        client_message="Во сколько это обойдётся за месяц?",
        expected_slot="grade",
        known_slots={},
        facts={
            "unpk.online.5_8.monthly": "УНПК: для 5-8 класса онлайн помесячная оплата считается от годовой цены 59 000 ₽.",
            "unpk.online.9_11.monthly": "УНПК: для 9-11 класса онлайн помесячная оплата считается от годовой цены 69 900 ₽.",
        },
        fact_metadata={},
        missing_facts=("уточнить класс для помесячной оплаты",),
    ),
    TargetCase(
        case_id="T04_unpk_schedule_no_subject",
        brand="unpk",
        client_message="Когда занятия?",
        expected_slot="subject",
        known_slots={"grade": "8", "format": "онлайн"},
        facts={
            "unpk.math.8.online.schedule": "УНПК: математика 8 класс онлайн проходит по вторникам 18:00-20:00.",
            "unpk.physics.8.online.schedule": "УНПК: физика 8 класс онлайн проходит по четвергам 19:00-21:00.",
        },
        fact_metadata={},
    ),
    TargetCase(
        case_id="T05_foton_schedule_no_subject",
        brand="foton",
        client_message="По каким дням проходят занятия?",
        expected_slot="subject",
        known_slots={"grade": "9", "format": "очно"},
        facts={
            "foton.math.9.offline.schedule": "Фотон: математика 9 класс очно проходит по субботам 12:00-14:00.",
            "foton.physics.9.offline.schedule": "Фотон: физика 9 класс очно проходит по воскресеньям 15:00-17:00.",
        },
        fact_metadata={},
    ),
    TargetCase(
        case_id="T06_unpk_discounts_no_format",
        brand="unpk",
        client_message="А скидки есть?",
        expected_slot="format",
        known_slots={"grade": "8", "subject": "физика"},
        facts={
            "unpk.offline.discount": "УНПК: для очного формата действует скидка 10% при оплате года.",
            "unpk.online.discount": "УНПК: для онлайн-формата действует скидка 7% при оплате года.",
        },
        fact_metadata={},
        missing_facts=("уточнить формат для скидки",),
    ),
    TargetCase(
        case_id="T07_foton_group_no_format",
        brand="foton",
        client_message="Какая группа подойдёт?",
        expected_slot="format",
        known_slots={"grade": "8", "subject": "математика"},
        facts={
            "foton.math.8.offline.group": "Фотон: очная группа математики 8 класса занимается в Москве.",
            "foton.math.8.online.group": "Фотон: онлайн-группа математики 8 класса занимается в Zoom.",
        },
        fact_metadata={},
    ),
    TargetCase(
        case_id="T08_unpk_group_no_time",
        brand="unpk",
        client_message="Подберите группу по физике онлайн.",
        expected_slot="time",
        known_slots={"grade": "8", "subject": "физика", "format": "онлайн"},
        facts={
            "unpk.physics.8.online.weekday": "УНПК: физика 8 класс онлайн по будням идёт во вторник 18:00-20:00.",
            "unpk.physics.8.online.weekend": "УНПК: физика 8 класс онлайн по выходным идёт в воскресенье 14:30-16:30.",
        },
        fact_metadata={},
        missing_facts=("уточнить удобное время",),
    ),
    TargetCase(
        case_id="T09_foton_group_no_time",
        brand="foton",
        client_message="Хочу выбрать группу по математике.",
        expected_slot="time",
        known_slots={"grade": "9", "subject": "математика", "format": "очно"},
        facts={
            "foton.math.9.offline.weekday": "Фотон: математика 9 класс очно по будням идёт в среду 18:30-20:30.",
            "foton.math.9.offline.weekend": "Фотон: математика 9 класс очно по выходным идёт в субботу 12:00-14:00.",
        },
        fact_metadata={},
        missing_facts=("уточнить удобное время",),
    ),
    TargetCase(
        case_id="T10_unpk_course_no_subject",
        brand="unpk",
        client_message="Какой курс есть для подготовки?",
        expected_slot="subject",
        known_slots={"grade": "11", "format": "онлайн"},
        facts={
            "unpk.math.11.online.ege": "УНПК: математика 11 класс онлайн готовит к ЕГЭ по понедельникам.",
            "unpk.physics.11.online.ege": "УНПК: физика 11 класс онлайн готовит к ЕГЭ по средам.",
        },
        fact_metadata={},
    ),
)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.parallel) as executor:
        futures = {executor.submit(run_case_pair, case): case for case in TARGET_CASES}
        for future in as_completed(futures):
            rows.append(future.result())
    rows.sort(key=lambda row: row["case_id"])

    write_jsonl(out_dir / "target_question_rows.jsonl", rows)
    (out_dir / "transcripts.md").write_text(render_transcripts(rows), encoding="utf-8")
    summary = build_summary(rows, out_dir=out_dir, parallel=args.parallel)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["gate_passed"] else 2


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run target TZ123 question-instead-of-handoff cases.")
    parser.add_argument("--out-dir", type=Path, default=Path("audits/_inbox/tz123_target_question_set_20260616"))
    parser.add_argument("--parallel", type=int, default=4)
    args = parser.parse_args(argv)
    if args.parallel < 1:
        raise SystemExit("--parallel must be >= 1")
    return args


def run_case_pair(case: TargetCase) -> dict[str, Any]:
    with target_env(question_enabled=False):
        off = run_case(case, question_enabled=False)
    with target_env(question_enabled=True):
        on = run_case(case, question_enabled=True)
    return {
        "case_id": case.case_id,
        "brand": case.brand,
        "client_message": case.client_message,
        "expected_slot": case.expected_slot,
        "off": off,
        "on": on,
        "checks": checks_for_pair(case, off=off, on=on),
    }


def run_case(case: TargetCase, *, question_enabled: bool) -> dict[str, Any]:
    result = make_handoff_result(case)
    context = make_context(case)
    output = apply_question_instead_of_handoff_layer(
        result,
        client_message=case.client_message,
        context=context if question_enabled else {**context, QUESTION_INSTEAD_OF_HANDOFF_ENV: "0"},
    )
    meta = output.metadata.get("question_instead_of_handoff") if isinstance(output.metadata, Mapping) else {}
    action_decision = output.metadata.get("action_decision") if isinstance(output.metadata, Mapping) else {}
    return {
        "route": output.route,
        "draft_text": output.draft_text,
        "safety_flags": list(output.safety_flags),
        "manager_followup_required": output.manager_followup_required,
        "question_meta": dict(meta) if isinstance(meta, Mapping) else {},
        "action_decision": dict(action_decision) if isinstance(action_decision, Mapping) else {},
    }


def make_handoff_result(case: TargetCase) -> SubscriptionDraftResult:
    return SubscriptionDraftResult(
        message_type="question",
        topic_id="theme:001_pricing" if case.expected_slot in {"grade", "format"} else "theme:schedule",
        risk_level="low",
        route="draft_for_manager",
        draft_text="Передам вопрос менеджеру, он уточнит детали и сориентирует.",
        missing_facts=case.missing_facts,
        safety_flags=("manager_approval_required", "no_auto_send", "draft_only", "direct_path_model"),
        metadata={
            "authoritative_output_gate": {"action": "pass", "findings": []},
            "action_decision": {
                "schema_version": "deal_action_decision_v1_2026_06_14",
                "enabled": True,
                "action": "answer_only",
                "confidence": 1.0,
                "reason": "target_question_set",
                "requires_manager_approval": True,
                "no_live_execution": True,
            },
            "direct_path": {
                "active_brand": case.brand,
                "retrieved_facts": dict(case.facts),
                "wide_fact_exact_keys": list(case.facts.keys()),
                "wide_fact_metadata": dict(case.fact_metadata),
            },
        },
    )


def make_context(case: TargetCase) -> dict[str, Any]:
    memory = build_dialogue_memory(
        current_message=case.client_message,
        active_brand=case.brand,
        recent_messages=(f"Клиент: {case.client_message}",),
        known_slots=case.known_slots,
        session_id=f"tz123_target:{case.case_id}",
    ).to_prompt_view()
    return {
        "active_brand": case.brand,
        DIRECT_PATH_ENV: "1",
        DIRECT_PATH_PILOT_CONFIG_ENV: DIRECT_PATH_PILOT_CONFIG_VERSION,
        QUESTION_INSTEAD_OF_HANDOFF_ENV: "1",
        "known_slots": dict(case.known_slots),
        "known_dialog_fields": dict(case.known_slots),
        "dialogue_memory_view": memory,
        "recent_messages": [f"Клиент: {case.client_message}"],
    }


def checks_for_pair(case: TargetCase, *, off: Mapping[str, Any], on: Mapping[str, Any]) -> dict[str, bool]:
    on_meta = on.get("question_meta") if isinstance(on.get("question_meta"), Mapping) else {}
    on_text = str(on.get("draft_text") or "")
    lower = on_text.casefold().replace("ё", "е")
    slot_mentions = {
        "grade": "класс" in lower,
        "subject": "предмет" in lower,
        "format": ("формат" in lower and ("очно" in lower or "онлайн" in lower)),
        "time": "время" in lower,
    }
    return {
        "off_would_handoff": str(off.get("route") or "") == "draft_for_manager",
        "off_requires_manager": "manager_approval_required" in set(off.get("safety_flags") or []),
        "on_fired": on_meta.get("status") == "fired",
        "slot_expected": on_meta.get("slot") == case.expected_slot,
        "on_self_route": str(on.get("route") or "") == "bot_answer_self_for_pilot",
        "one_human_question": is_one_human_question(on_text),
        "only_expected_slot_question": slot_mentions.get(case.expected_slot, False)
        and sum(1 for value in slot_mentions.values() if value) == 1,
        "not_p0_input": not P0_RE.search(case.client_message),
        "brand_clean": brand_clean(case.brand, on_text),
        "no_loop": "ещё" not in lower and "если уже" not in lower and lower.count("подскажите") <= 1,
    }


def is_one_human_question(text: str) -> bool:
    lower = str(text or "").casefold()
    if not lower.startswith("подскажите, пожалуйста"):
        return False
    if any(marker in lower for marker in ("менеджер", "передам", "заявк", "crm", "id", "{", "}")):
        return False
    slot_markers = sum(1 for marker in ("класс", "предмет", "формат", "время") if marker in lower)
    return slot_markers == 1


def brand_clean(brand: str, text: str) -> bool:
    lower = str(text or "").casefold().replace("ё", "е")
    if brand == "unpk":
        return "фотон" not in lower
    if brand == "foton":
        return "унпк" not in lower and "мфти" not in lower
    return False


def build_summary(rows: Sequence[Mapping[str, Any]], *, out_dir: Path, parallel: int) -> dict[str, Any]:
    failed = [
        {"case_id": row["case_id"], "check": name}
        for row in rows
        for name, passed in dict(row.get("checks") or {}).items()
        if not passed
    ]
    fired_rows = [
        row for row in rows if ((row.get("on") or {}).get("question_meta") or {}).get("status") == "fired"
    ]
    return {
        "schema_version": TARGET_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "out_dir": str(out_dir),
        "parallel": parallel,
        "cases_total": len(rows),
        "off_handoff": sum(1 for row in rows if (row.get("off") or {}).get("route") == "draft_for_manager"),
        "on_fired": len(fired_rows),
        "on_by_slot": dict(Counter(str(((row.get("on") or {}).get("question_meta") or {}).get("slot") or "") for row in fired_rows)),
        "failed_checks": failed,
        "gate_passed": len(rows) in range(8, 11) and len(fired_rows) > 0 and not failed,
        "llm_calls_total": 0,
        "safety": {
            "writes_crm": False,
            "writes_tallanto": False,
            "writes_amo": False,
            "sends_messages": False,
            "runs_asr": False,
            "touches_stable_runtime": False,
        },
        "semantic_review": {
            "verdict": "PASS_WITH_NOTES" if not failed else "BLOCKED",
            "notes": [
                "Target harness checks the deterministic post-layer on draft_for_manager+answer_only inputs.",
                "Full Codex direct-path replay is covered separately by TZ123+TZ124 remainder measure.",
            ],
        },
    }


@contextmanager
def target_env(*, question_enabled: bool) -> Iterable[None]:
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


def render_transcripts(rows: Sequence[Mapping[str, Any]]) -> str:
    lines = ["# TZ-123 Target Question Set", ""]
    for row in rows:
        lines.extend(
            [
                f"## {row['case_id']} ({row['brand']})",
                "",
                f"CLIENT: {row['client_message']}",
                f"OFF_ROUTE: {(row.get('off') or {}).get('route')}",
                f"ON_ROUTE: {(row.get('on') or {}).get('route')}",
                f"ON_SLOT: {((row.get('on') or {}).get('question_meta') or {}).get('slot', '')}",
                f"ON_TEXT: {(row.get('on') or {}).get('draft_text')}",
                f"CHECKS: {json.dumps(row.get('checks') or {}, ensure_ascii=False, sort_keys=True)}",
                "",
            ]
        )
    return "\n".join(lines)


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
