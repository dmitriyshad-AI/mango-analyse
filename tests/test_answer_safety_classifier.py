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
    codes_from_text,
)
from mango_mvp.channels.subscription_llm_parts.support import _p0_model_led_filter_high_risk_codes


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


@pytest.mark.parametrize(
    ("message", "expected_code"),
    (
        ("Хочу снять ребёнка с кружка.", "refund"),
        ("Можно отказаться и выписать ребёнка с занятий?", "refund"),
        ("Нужно перенести оплаченную смену, или вернёте деньги за смену?", "refund"),
        ("В договоре неверная дата и фамилия ребёнка, исправьте.", "legal"),
    ),
)
def test_p0_model_led_preserves_three_class_codes(message: str, expected_code: str) -> None:
    codes = codes_from_text(message)

    assert expected_code in codes
    assert _p0_model_led_filter_high_risk_codes(codes, client_message=message, context={}) == codes


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


@pytest.mark.parametrize(
    ("message", "expected_code"),
    (
        ("Оплатил онлайн-математику — доступа так и нет.", "payment_dispute"),
        ("Внесли оплату, приглашение не пришло.", "payment_dispute"),
        ("Оплатили курс, логин и пароль не дали.", "payment_dispute"),
        ("Оплатил, ссылка на платформу не пришла.", "payment_dispute"),
        ("Оплату внесли, а приглашение на онлайн физику для 9 класса так и не пришло.", "payment_dispute"),
        ("Оплатили, ссылка на платформу для онлайн занятий до сих пор не пришла.", "payment_dispute"),
        ("Оплачен курс, логин и пароль в личном кабинете так и не дали.", "payment_dispute"),
        ("Оплатили курс, доступ к личному кабинету для онлайн физики не открыли.", "payment_dispute"),
        ("Преподаватель не объясняет, ребёнок ничего не понимает.", "complaint"),
        ("Педагог некомпетентный, ничему не учит.", "complaint"),
        ("Это безобразие, как ведут занятия.", "complaint"),
    ),
)
def test_tz145_p0_detector_covers_payment_access_and_quality_complaints(message: str, expected_code: str) -> None:
    decision = classify_answer_safety(client_message=message)

    assert expected_code in decision.risk_codes
    assert decision.p0_required is True
    assert decision.manager_only is True


@pytest.mark.parametrize(
    "message",
    (
        "Как выбрать преподавателя?",
        "Ребёнок стесняется отвечать при всех, что посоветуете?",
        "Оплачу позже.",
        "Оплатил два курса.",
        "Оплатила вчера, занятия завтра — в системе пока нет.",
    ),
)
def test_tz145_p0_detector_keeps_benign_controls_non_p0(message: str) -> None:
    decision = classify_answer_safety(client_message=message)

    assert codes_from_current_message(message) == ()
    assert decision.p0_required is False
    assert decision.manager_only is False


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


def test_answer_safety_post_payment_refund_process_question_is_p0() -> None:
    """D-087 defect: prior client turn confirms payment ("я уже оплатил"), current
    turn asks the refund process question with presale-shaped wording ("как
    оформить возврат?"). In isolation the current message matches the benign
    presale process frame, but the recent-turn haystack already carries the
    REFUND_POST_PAYMENT structural signal ("оплатил") — that must win."""
    decision = classify_answer_safety(
        client_message="Как оформить возврат?",
        context={
            "recent_messages": [
                "Клиент: Я уже оплатил.",
                "Ответ: Отлично, доступ откроем к началу занятий.",
            ],
        },
    )

    assert "refund" in decision.risk_codes
    assert decision.primary_risk == "refund"
    assert decision.p0_required is True
    assert decision.manager_only is True
    assert decision.zero_collect_required is True
    assert decision.blocks_autonomy is True


def test_answer_safety_post_payment_refund_cannot_be_downgraded_by_presale_plan() -> None:
    decision = classify_answer_safety(
        client_message="Как оформить возврат?",
        context={
            "recent_messages": ["Клиент: Я уже оплатил.", "Ответ: Хорошо."],
            "conversation_intent_plan": {
                "primary_intent": "refund",
                "refund_frame": "presale_policy",
                "risk_signals": [],
            },
        },
    )

    assert decision.semantic_non_p0 is False
    assert decision.p0_required is True
    assert decision.manager_only is True
    assert decision.risk_codes == ("refund",)


@pytest.mark.parametrize(
    ("prev_client_turn", "current_message"),
    (
        ("Клиент: Я уже оплатил.", "Как оформить возврат?"),
        ("Клиент: Оплатила курс вчера.", "Какая процедура возврата?"),
        ("Клиент: Мы оплатили обучение.", "Нужно писать заявление на возврат?"),
        ("Клиент: С меня списали оплату за курс.", "Подскажите порядок возврата."),
        ("Клиент: Оплатил, верните деньги.", "Оплатил, верните деньги."),
    ),
)
def test_answer_safety_post_payment_refund_class_stays_p0(prev_client_turn: str, current_message: str) -> None:
    decision = classify_answer_safety(
        client_message=current_message,
        context={"recent_messages": [prev_client_turn, "Ответ: Хорошо."]},
    )

    assert decision.p0_required is True
    assert decision.manager_only is True
    assert decision.zero_collect_required is True


@pytest.mark.parametrize(
    "message",
    (
        "До оплаты хочу понять условия возврата.",
        "Если оплачу и передумаю до начала, деньги вернут?",
    ),
)
def test_answer_safety_presale_refund_benign_exception_survives_post_payment_fix(message: str) -> None:
    """The two benign pre-sale exceptions named in the D-087 fix spec must stay non-P0."""
    decision = classify_answer_safety(client_message=message)

    assert decision.p0_required is False
    assert decision.manager_only is False
    assert decision.zero_collect_required is False


def test_answer_safety_presale_refund_with_unrelated_recent_context_stays_non_p0() -> None:
    """A haystack that mentions something else entirely (no payment signal) must
    not be swept into P0 just because recent_messages is non-empty."""
    decision = classify_answer_safety(
        client_message="Перед оплатой хочу понять условия возврата.",
        context={
            "recent_messages": [
                "Клиент: А какое расписание по субботам?",
                "Ответ: Занятия по субботам в 10:00.",
            ],
        },
    )

    assert decision.risk_codes == ()
    assert decision.p0_required is False
    assert decision.manager_only is False


def test_answer_safety_active_payment_dispute_latch_survives_post_payment_presale_wording() -> None:
    """An already-active hard P0 latch (payment_dispute) must not be lifted just
    because a later client turn re-asks the refund question in presale-shaped
    wording — the active latch/dispute is not cleared by D-087."""
    decision = classify_answer_safety(
        client_message="Как оформить возврат?",
        context={
            "recent_messages": [
                "Клиент: Я оплатил, но в системе нет моего платежа, деньги списали!",
                "Ответ: Приняли вопрос по оплате. Передам его менеджеру.",
            ],
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
    assert decision.manager_only is True
    assert "payment_dispute" in decision.risk_codes


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
                    "had_hard_p0_claim": False,
                }
            },
        },
    )

    assert decision.p0_required is False
    assert decision.zero_collect_required is False
    assert decision.risk_codes == ()


def test_answer_safety_presale_wording_cannot_release_hard_refund_latch() -> None:
    decision = classify_answer_safety(
        client_message="В целом, просто заранее спрашиваю: если передумаем, вернут остаток?",
        context={
            "recent_messages": ["Клиент: Подскажите расписание.", "Ответ: Сейчас расскажу."],
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

    assert decision.p0_required is True
    assert decision.manager_only is True
    assert decision.risk_codes == ("refund",)


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
