from __future__ import annotations

import re
from pathlib import Path

import pytest

from mango_mvp.channels.answer_safety_classifier import classify_answer_safety, codes_from_current_message
from mango_mvp.channels.p0_recall_spec import (
    PAYMENT_DISPUTE_BENIGN_CASES,
    PAYMENT_DISPUTE_POSITIVE_CASES,
    PAYMENT_DISPUTE_RE,
    P0_BENIGN_CASES,
    P0_TRUE_POSITIVE_CASES,
)


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


@pytest.mark.parametrize("message", PAYMENT_DISPUTE_POSITIVE_CASES)
def test_payment_dispute_positive_corpus_matches_runtime_regex(message: str) -> None:
    decision = classify_answer_safety(client_message=message)

    assert PAYMENT_DISPUTE_RE.search(message)
    assert "payment_dispute" in decision.risk_codes
    assert decision.p0_required is True


@pytest.mark.parametrize("message", PAYMENT_DISPUTE_BENIGN_CASES)
def test_payment_dispute_benign_corpus_does_not_match_runtime_regex(message: str) -> None:
    decision = classify_answer_safety(client_message=message)

    assert PAYMENT_DISPUTE_RE.search(message) is None
    assert "payment_dispute" not in decision.risk_codes
    assert decision.p0_required is False


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
        "Если ребёнок надолго заболеет, за пропущенное вернёте?",
        "До оплаты хочу понимать правила возврата, это не жалоба.",
        "В целом, без договора, просто спрашиваю: если передумаем, вернут остаток?",
        "Гипотетически, до оплаты, если уже начнём и поймём, что формат не подходит, возврат возможен?",
        "Если оплачу и до начала занятий передумаю, деньги вернут?",
        "Поняла, но именно про возврат можете уточнить? Это не жалоба, просто хочу заранее понимать правила до оплаты.",
    ):
        decision = classify_answer_safety(client_message=message)

        assert codes_from_current_message(message) == ()
        assert decision.p0_required is False
        assert decision.manager_only is False


def test_answer_safety_presale_refund_followup_overrides_stale_refund_context_and_latch() -> None:
    decision = classify_answer_safety(
        client_message="В целом, без договора, просто спрашиваю: если передумаем, вернут остаток?",
        context={
            "recent_messages": [
                "Клиент: если передумаем до начала, деньги вернут?",
                "Бот: возвращается остаток неистраченных средств.",
            ],
            "dialogue_memory_view": {
                "p0_latch": {
                    "active": True,
                    "codes": ["refund"],
                    "primary_risk": "refund",
                }
            },
        },
    )

    assert decision.p0_required is False
    assert decision.manager_only is False
    assert "refund" not in decision.risk_codes
    assert decision.semantic_non_p0 is True


def test_answer_safety_presale_refund_latch_does_not_leak_to_neutral_followup() -> None:
    decision = classify_answer_safety(
        client_message="Понял, спасибо. Посмотрю программу и расписание",
        context={
            "recent_messages": [
                "Клиент: А если не подойдёт, можно будет вернуть деньги?",
                "Ответ: Да, при досрочном отказе возвращается остаток неистраченных средств.",
            ],
            "conversation_intent_plan": {
                "primary_intent": "schedule",
                "risk_signals": [],
                "route_bias": "bot_answer_self_for_pilot",
            },
            "dialogue_memory_view": {
                "p0_latch": {
                    "active": True,
                    "codes": ["refund"],
                    "primary_risk": "refund",
                    "had_hard_p0_claim": True,
                }
            },
        },
    )

    assert decision.p0_required is False
    assert decision.zero_collect_required is False
    assert decision.risk_codes == ()


def test_answer_safety_presale_context_does_not_release_payment_dispute_latch() -> None:
    decision = classify_answer_safety(
        client_message="Понял, спасибо. Посмотрю программу и расписание",
        context={
            "recent_messages": [
                "Клиент: А если не подойдёт, можно будет вернуть деньги?",
                "Ответ: Да, при досрочном отказе возвращается остаток неистраченных средств.",
            ],
            "conversation_intent_plan": {
                "primary_intent": "schedule",
                "risk_signals": [],
                "route_bias": "bot_answer_self_for_pilot",
            },
            "dialogue_memory_view": {
                "p0_latch": {
                    "active": True,
                    "codes": ["payment_dispute"],
                    "primary_risk": "payment_dispute",
                    "had_hard_p0_claim": True,
                }
            },
        },
    )

    assert decision.p0_required is True
    assert decision.primary_risk == "payment_dispute"
    assert "payment_dispute" in decision.risk_codes


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


@pytest.mark.parametrize(
    "message",
    (
        "Я оплатил информатику, занятий нет, верните деньги.",
        "Верните деньги.",
        "Списали дважды, верните лишний платёж.",
        "Буду писать претензию и пойду в суд.",
    ),
)
def test_answer_safety_real_refund_or_legal_claims_stay_p0(message: str) -> None:
    decision = classify_answer_safety(client_message=message)

    assert decision.p0_required is True
    assert decision.manager_only is True


def test_answer_safety_soft_reputation_marker_does_not_force_p0() -> None:
    decision = classify_answer_safety(client_message="Я видел отзывы в интернете, вас точно не обманывают?")

    assert decision.p0_required is False
    assert decision.manager_only is False
    assert decision.blocks_autonomy is False


def test_answer_safety_reputation_threat_is_soft_marker_not_hard_p0() -> None:
    decision = classify_answer_safety(client_message="Напишу отзыв в интернете, если не подскажете условия.")

    assert "reputation_threat" in decision.risk_codes
    assert decision.p0_required is False
    assert decision.manager_only is False


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
