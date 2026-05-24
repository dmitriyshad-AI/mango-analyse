from __future__ import annotations

import re
from pathlib import Path

import pytest

from mango_mvp.channels.answer_safety_classifier import classify_answer_safety, codes_from_current_message
from mango_mvp.channels.p0_recall_spec import P0_BENIGN_CASES, P0_TRUE_POSITIVE_CASES


@pytest.mark.parametrize(("message", "expected_code"), P0_TRUE_POSITIVE_CASES)
def test_answer_safety_real_p0_recall_matrix_requires_manager_only(message: str, expected_code: str) -> None:
    decision = classify_answer_safety(client_message=message)

    assert expected_code in decision.risk_codes
    assert decision.p0_required is True
    assert decision.manager_only is True
    assert decision.blocks_autonomy is True
    assert decision.blocks_rewriter is True


@pytest.mark.parametrize("message", P0_BENIGN_CASES)
def test_answer_safety_benign_process_phrases_do_not_require_p0(message: str) -> None:
    decision = classify_answer_safety(client_message=message)

    assert codes_from_current_message(message) == ()
    assert decision.p0_required is False
    assert decision.manager_only is False
    assert decision.blocks_autonomy is False


def test_answer_safety_active_p0_latch_blocks_semantic_non_p0_repair() -> None:
    decision = classify_answer_safety(
        client_message="А теперь скажите цену на год.",
        context={
            "conversation_intent_plan": {
                "primary_intent": "pricing",
                "risk_signals": [],
                "route_bias": "bot_answer_self_for_pilot",
            },
            "dialogue_memory_view": {
                "p0_latch": {
                    "active": True,
                    "codes": ["payment_dispute"],
                    "primary_risk": "payment_dispute",
                }
            },
        },
    )

    assert "payment_dispute" in decision.risk_codes
    assert decision.p0_required is True
    assert decision.manager_only is True
    assert decision.semantic_non_p0 is False


def test_answer_safety_presale_refund_policy_question_is_not_full_p0() -> None:
    for message in (
        "А если ребёнку не понравится, деньги вернёте?",
        "Перед оплатой хочу понять условия возврата.",
        "До оплаты хочу понимать правила возврата, это не жалоба.",
        "Поняла, но именно про возврат можете уточнить? Это не жалоба, просто хочу заранее понимать правила до оплаты.",
    ):
        decision = classify_answer_safety(client_message=message)

        assert codes_from_current_message(message) == ()
        assert decision.p0_required is False
        assert decision.manager_only is False


def test_answer_safety_presale_refund_repairs_wrong_refund_topic() -> None:
    decision = classify_answer_safety(
        client_message="До оплаты хочу понять условия возврата.",
        topic_id="theme:009_refund",
        context={
            "conversation_intent_plan": {
                "primary_intent": "general_consultation",
                "risk_signals": [],
                "route_bias": "draft_for_manager",
            }
        },
    )

    assert decision.p0_required is False
    assert decision.risk_codes == ()
    assert decision.semantic_non_p0 is True


def test_answer_safety_active_refund_request_stays_p0() -> None:
    decision = classify_answer_safety(client_message="Мы уже оплатили курс, ребёнку не понравилось, верните деньги.")

    assert "refund" in decision.risk_codes
    assert decision.p0_required is True
    assert decision.manager_only is True


def test_p0_text_regexes_live_only_in_p0_recall_spec() -> None:
    channels_dir = Path(__file__).resolve().parents[1] / "src" / "mango_mvp" / "channels"
    forbidden_defs = re.compile(
        r"\b(?:REFUND_RE|LEGAL_RE|COMPLAINT_RE|PAYMENT_DISPUTE_RE|P0_TEXT_RE|P0_MARKERS)\s*="
    )
    forbidden_helpers = (
        "def _has_refund_signal",
        "def _has_legal_signal",
        "def _has_complaint_signal",
    )

    offenders: list[str] = []
    for path in channels_dir.glob("*.py"):
        if path.name == "p0_recall_spec.py":
            continue
        text = path.read_text(encoding="utf-8")
        if forbidden_defs.search(text) or any(marker in text for marker in forbidden_helpers):
            offenders.append(path.name)

    assert offenders == []
