from __future__ import annotations

import json
from pathlib import Path

from scripts import report_adr003_partial_answer_opportunities as report


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def _fact(
    *,
    brand: str = "foton",
    fact_key: str = "price.online.semester",
    fact_type: str = "price",
    product: str = "regular_course",
    program_kind: str = "regular",
    text: str = "Фотон: онлайн, семестр — 29 750 ₽.",
    structured_value: dict | None = None,
) -> dict:
    return {
        "allowed_for_client_answer": True,
        "brand": brand,
        "client_safe_text": text,
        "fact_id": f"fact:{brand}:{fact_key}",
        "fact_key": fact_key,
        "fact_type": fact_type,
        "forbidden_for_client": False,
        "internal_only": False,
        "product": product,
        "program_kind": program_kind,
        "structured_value": structured_value or {"valid_until": "2026-12-31"},
        "valid_until": "2026-12-31",
    }


def _turn(
    dialog_id: str,
    *,
    route: str = "draft_for_manager",
    brand: str = "foton",
    action: str = "answer_question",
    risk_class: str = "missing_facts",
    payment_readiness: str = "none",
    deal_stage: str = "research",
    missing_facts: list[str] | None = None,
    requested_product: dict | None = None,
) -> dict:
    return {
        "brand": brand,
        "dialog_id": dialog_id,
        "turns": [
            {
                "bot_missing_facts": missing_facts or ["platform.current", "класс ребёнка для точной цены"],
                "bot_route": route,
                "bot_semantic_frame": {
                    "answerability": "manager_only",
                    "confidence": 0.88,
                    "deal_stage": deal_stage,
                    "must_handoff": True,
                    "payment_readiness": payment_readiness,
                    "requested_action": action,
                    "requested_product": requested_product
                    or {
                        "brand": brand,
                        "format": "онлайн",
                        "grade": "",
                        "program_kind": "regular_course",
                        "raw_text": "онлайн семестр",
                        "subject": "",
                    },
                    "risk_class": risk_class,
                },
                "client_message": "Сколько стоит онлайн и где он проходит?",
                "turn": 1,
            }
        ],
    }


def _build(tmp_path: Path, *, dialogs: list[dict], facts: list[dict]) -> dict:
    transcripts = tmp_path / "transcripts.jsonl"
    kb = tmp_path / "kb.json"
    _write_jsonl(transcripts, dialogs)
    _write_json(kb, {"facts": facts})
    return report.build_report(transcripts=transcripts, kb_snapshot=kb)


def test_draft_partial_shadow_candidate_is_report_only(tmp_path: Path) -> None:
    result = _build(
        tmp_path,
        dialogs=[_turn("partial")],
        facts=[
            _fact(),
            _fact(
                fact_key="online_platform_transition",
                fact_type="process",
                text="С лета 2026 онлайн-занятия проходят на платформе SohoLMS.",
            ),
        ],
    )

    case = result["partial_cases"][0]
    assert case["partial_answer_shadow"]["status"] == "draft_partial_shadow_candidate"
    assert case["partial_answer_shadow"]["active_behavior_allowed"] is False
    assert case["partial_answer_shadow"]["generated_text_exported"] is False
    assert case["kb_support"]["proven_parts"] == ["price_cost", "platform_current"]
    assert case["kb_support"]["missing_slots"] == ["grade"]
    assert result["totals"]["draft_partial_shadow_candidates"] == 1
    dumped = json.dumps(result, ensure_ascii=False)
    assert "29 750" not in dumped
    assert "SohoLMS" not in dumped


def test_manager_only_partial_support_stays_policy_blocked(tmp_path: Path) -> None:
    result = _build(
        tmp_path,
        dialogs=[_turn("manager", route="manager_only")],
        facts=[_fact()],
    )

    case = result["partial_cases"][0]
    assert case["partial_answer_shadow"]["status"] == "manager_only_policy_blocked"
    assert "route_manager_only" in case["partial_answer_shadow"]["why_not_active"]
    assert result["totals"]["manager_only_partial_policy_blocked"] == 1


def test_live_availability_missing_axis_blocks_partial_candidate(tmp_path: Path) -> None:
    result = _build(
        tmp_path,
        dialogs=[
            _turn(
                "availability",
                missing_facts=["platform.current", "актуальное наличие мест"],
            )
        ],
        facts=[
            _fact(
                fact_key="online_platform_transition",
                fact_type="process",
                text="С лета 2026 онлайн-занятия проходят на платформе SohoLMS.",
            )
        ],
    )

    case = result["partial_cases"][0]
    assert case["partial_answer_shadow"]["status"] == "hard_missing_axis_blocked"
    assert "live_availability" in case["kb_support"]["uncovered_categories"]


def test_money_or_enroll_actions_are_excluded_even_with_partial_support(tmp_path: Path) -> None:
    result = _build(
        tmp_path,
        dialogs=[
            _turn(
                "money",
                action="send_payment_link",
                payment_readiness="ready_to_pay",
            )
        ],
        facts=[_fact()],
    )

    case = result["partial_cases"][0]
    assert case["partial_answer_shadow"]["status"] == "action_or_danger_excluded"
    assert result["totals"]["action_or_danger_excluded_partial_rows"] == 1


def test_camp_price_is_not_partial_support_for_online_regular_course(tmp_path: Path) -> None:
    result = _build(
        tmp_path,
        dialogs=[_turn("camp-price")],
        facts=[
            _fact(
                fact_key="lvsh.price",
                product="camp",
                program_kind="camp",
                text="Фотон: ЛВШ Менделеево, текущая цена — 114 000 ₽.",
            )
        ],
    )

    assert result["partial_cases"] == []
    assert result["totals"]["partial_support_handoff_turns"] == 0
