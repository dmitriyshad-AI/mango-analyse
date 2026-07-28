from __future__ import annotations

import itertools
import re
from pathlib import Path

import pytest

from mango_mvp.channels.answer_safety_classifier import classify_answer_safety, codes_from_current_message
from mango_mvp.channels.output_verification_floor import p0_pre_gate
from mango_mvp.channels.p0_recall_spec import (
    PAYMENT_DISPUTE_BENIGN_CASES,
    PAYMENT_DISPUTE_POSITIVE_CASES,
    PAYMENT_DISPUTE_RE,
    P0_BENIGN_CASES,
    P0_TRUE_POSITIVE_CASES,
    codes_from_text,
    hard_codes_from_text,
    is_benign_hypothetical_refund,
)
from mango_mvp.channels.semantic_roles import (
    REFUND_POST_PAYMENT,
    REFUND_PRESALE_FRAME,
    _refund_frame,
    has_post_payment_refund_evidence,
    is_negated_refund_topic,
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
    for path in channels_dir.rglob("*.py"):
        if path.name == "p0_recall_spec.py":
            continue
        text = path.read_text(encoding="utf-8")
        if forbidden_defs.search(text) or any(marker in text for marker in forbidden_helpers):
            offenders.append(path.name)

    assert offenders == []


def test_d1_top_risk_post_payment_refund_is_never_downgraded_to_presale_policy() -> None:
    """D-1 top-risk fix (owner audit 2026-07-26): before the fix, semantic_roles.
    _refund_frame guarded its two presale branches with a hand-copied, narrower
    duplicate of REFUND_POST_PAYMENT -- ("уже оплат", "оплатил", "оплатила",
    "списали", "сняли") -- that missed "мы платили", "я платил", "после оплаты",
    "заключили договор", "за наш" and bare "списал". A genuine post-payment refund
    claim phrased with one of the missed markers plus any REFUND_PRESALE_FRAME word
    (e.g. "если") was silently reclassified as presale_policy. That made
    p0_recall_spec.codes_from_text drop the "refund" code and let
    output_verification_floor.p0_pre_gate -- the deterministic gate the live
    direct-path build_draft() calls before ever running the model -- return None: a
    real refund claim after payment could have been answered by the bot instead of
    handed to a human manager."""
    for message in (
        "Мы платили за смену в лагере. Если вдруг что-то не устроит, деньги вернёте?",
        "Я платил за курс, а если ребёнку не понравится, вернёте?",
        "Мы уже заключили договор, а если передумаем, деньги вернёте?",
    ):
        frame, evidence = _refund_frame(message)
        assert frame == "dispute", f"{message!r} -> refund_frame={frame!r} ({evidence}), expected dispute"

        codes = codes_from_text(message)
        assert "refund" in codes, f"{message!r} -> codes_from_text={codes!r}, expected 'refund' present"
        assert "refund" in hard_codes_from_text(message)

        assert p0_pre_gate(message) is not None, f"{message!r} was not caught by the production P0 gate"


@pytest.mark.parametrize(
    ("paid_marker", "presale_marker"),
    list(itertools.product(REFUND_POST_PAYMENT, REFUND_PRESALE_FRAME)),
)
def test_d1_any_post_payment_marker_always_beats_any_presale_frame_marker(paid_marker: str, presale_marker: str) -> None:
    """Invariant (more important than the fix itself, per the D-1 audit): ANY
    REFUND_POST_PAYMENT marker combined with ANY REFUND_PRESALE_FRAME marker must
    still resolve to a hard P0 refund signal that reaches the production gate, never
    to the benign presale_policy frame. This is a full cross-product (11 x 29 = 319
    cases) over the two canonical marker tables, so it automatically re-covers any
    future addition to either list -- exactly the class of drift that caused the
    original defect (a hand-copied subset of REFUND_POST_PAYMENT silently fell out
    of sync with the real list). Money already moved must always dominate
    hypothetical/pre-sale wording: a deterministic layer may tag a message as
    presale, but must never let that tag delete a hard signal that also carries
    post-payment evidence."""
    message = f"{paid_marker}, {presale_marker}, вернёте деньги?"

    codes = codes_from_text(message)
    assert "refund" in codes, (
        f"{message!r} lost its refund P0 code -- a real post-payment refund claim "
        "could be handled by the bot instead of a human manager."
    )
    assert p0_pre_gate(message) is not None


def test_d1_negated_refund_topic_respects_explicit_topic_negation_even_with_post_payment_evidence() -> None:
    """Owner audit follow-up (Codex semantic review BLOCKED): this test used to assert
    the opposite -- that any REFUND_POST_PAYMENT marker anywhere in the message always
    overrides an explicit "не про возврат" negation. That was the over-widened state
    Codex's semantic sample audit flagged: "это не про возврат, мы платили, вопрос про
    расписание" (payment mentioned only as background context, the actual question is
    about schedule) was wrongly kept P0 and routed to a human. An explicit topic
    negation is a stronger, more deliberate signal than REFUND_PRESALE_FRAME
    hypothetical wording (which must still lose to payment evidence, see
    test_d1_any_post_payment_marker_always_beats_any_presale_frame_marker above) -- it
    denies the refund topic outright, so payment evidence elsewhere in the same
    message no longer suppresses it."""
    assert is_negated_refund_topic("Это не про возврат, я платил за курс, но хочу обсудить детали.") is True
    assert is_negated_refund_topic("Это не про возврат, мы уже заключили договор, а не понравилось.") is True
    assert is_negated_refund_topic("Это не про возврат, мы платили, вопрос про расписание.") is True
    # Unaffected control case (no payment evidence) must keep working as before.
    assert is_negated_refund_topic("Я не про возврат, я про то, где смотреть запись.") is True
    # An explicit demand phrase alongside the negation must still block it -- a client
    # cannot use "это не про возврат" to slip a real refund demand past the guard.
    assert is_negated_refund_topic("Это не про возврат, я платил за курс, но всё равно верните деньги.") is False


REFUND_TOPIC_NEGATION_PHRASES: tuple[str, ...] = (
    "это не про возврат",
    "не про возврат",
    "не о возврате",
    "это не возврат",
)


@pytest.mark.parametrize(
    ("paid_marker", "negation_phrase"),
    list(itertools.product(REFUND_POST_PAYMENT, REFUND_TOPIC_NEGATION_PHRASES)),
)
def test_d2_explicit_topic_negation_beats_any_post_payment_marker(paid_marker: str, negation_phrase: str) -> None:
    """Negation-axis counterpart to test_d1_any_post_payment_marker_always_beats_any_presale_frame_marker
    (owner audit follow-up, Codex semantic review BLOCKED): a REFUND_POST_PAYMENT
    marker can appear in a message purely as background context ("мы платили") while
    the client explicitly says the current question is *not* about a refund at all
    ("это не про возврат"). Unlike REFUND_PRESALE_FRAME hypothetical wording (D-1,
    still must lose to payment evidence, see the test above), an explicit topic
    negation means there is no refund topic to downgrade -- codes_from_text must not
    fire "refund" for any of these 11 x 4 = 44 combinations."""
    message = f"{negation_phrase}, {paid_marker}, вопрос про расписание."

    codes = codes_from_text(message)
    assert "refund" not in codes, (
        f"{message!r} wrongly kept its refund P0 code despite an explicit topic negation -- "
        "a client saying this is not about a refund must not be routed to a human as if it were."
    )


@pytest.mark.parametrize("paid_marker", REFUND_POST_PAYMENT)
def test_d2_real_refund_question_still_beats_post_payment_marker_without_negation(paid_marker: str) -> None:
    """Paired control for the negation-axis test above: dropping the negation phrase
    but keeping the same payment marker and a genuine refund question must still
    resolve to hard P0. This guards against the opposite regression -- a fix for the
    negation gap that overcorrects into silencing real post-payment refund claims."""
    message = f"{paid_marker}, а деньги вернёте?"

    codes = codes_from_text(message)
    assert "refund" in codes, f"{message!r} lost its refund P0 code without any topic negation present"
    assert p0_pre_gate(message) is not None


@pytest.mark.parametrize(
    "message",
    (
        "Это не про возврат, но почему вы не вернули деньги?",
        "Это не про возврат, когда вы вернете оплату?",
        "Это не про возврат, хочу, чтобы вы вернули оплату.",
        "Это не про возврат, прошу оформить возврат оплаченного курса.",
        "Это не про возврат, мне нужен возврат за курс.",
    ),
)
def test_d2_topic_negation_never_hides_a_second_real_refund_request(message: str) -> None:
    assert "refund" in codes_from_text(message)
    assert classify_answer_safety(client_message=message).manager_only is True
    assert p0_pre_gate(message) is not None


@pytest.mark.parametrize(
    "message",
    (
        "Это не про возврат, мы оплатили, но можно отменить обучение?",
        "Я не про возврат. Прошу отменить запись на курс.",
    ),
)
def test_topic_negation_does_not_hide_cancellation_p0(message: str) -> None:
    assert "refund" in codes_from_text(message)
    assert classify_answer_safety(client_message=message).manager_only is True
    assert p0_pre_gate(message) is not None


def test_d1_is_benign_hypothetical_refund_requires_no_post_payment_evidence() -> None:
    """is_benign_hypothetical_refund is the shared "presale evidence" primitive read
    by answer_safety_classifier.py, output_verification_floor.py and
    dialogue_memory.py to decide whether a refund signal/latch may be released. It
    must independently confirm has_post_payment_refund_evidence() is False, not just
    trust _refund_frame's own internal state -- so a future bug in _refund_frame
    cannot silently reopen this defect through any of its other consumers."""
    assert is_benign_hypothetical_refund("Мы платили за смену в лагере. Если вдруг что-то не устроит, деньги вернёте?") is False
    assert is_benign_hypothetical_refund("Перед оплатой хочу понять условия возврата.") is True


def test_d1_refund_downgrade_guard_cannot_reintroduce_narrow_post_payment_duplicate() -> None:
    """Structural regression guard for the D-1 fix, modeled on
    test_p0_text_regexes_live_only_in_p0_recall_spec: the presale/negated refund
    guards must key off the single canonical REFUND_POST_PAYMENT signal
    (semantic_roles.has_post_payment_refund_evidence / the paid_hit variable already
    computed from it), never a hand-copied, independently-drifting subset of it --
    that drift is exactly what let a real post-payment refund claim slip past the
    bot-vs-manager gate before this fix. No function anywhere in channels/ is
    allowed to reintroduce that literal narrow tuple."""
    channels_dir = Path(__file__).resolve().parents[1] / "src" / "mango_mvp" / "channels"
    forbidden = '"уже оплат", "оплатил", "оплатила", "списали", "сняли"'
    offenders = [path.name for path in channels_dir.rglob("*.py") if forbidden in path.read_text(encoding="utf-8")]

    assert offenders == []


def test_d1_has_post_payment_refund_evidence_covers_the_previously_missed_markers() -> None:
    """Every REFUND_POST_PAYMENT marker must be individually detected by the
    canonical helper -- this is the single signal both _refund_frame's paid_hit and
    the p0_recall_spec-level defense-in-depth guard rely on."""
    for marker in REFUND_POST_PAYMENT:
        assert has_post_payment_refund_evidence(marker) is True, marker
    assert has_post_payment_refund_evidence("ничего похожего тут нет") is False


@pytest.mark.parametrize(
    "message",
    (
        "Преподаватель вернул домашнюю работу.",
        "Курс ещё не оплачен, какие условия возврата?",
    ),
)
def test_ambiguous_non_payment_words_do_not_create_post_payment_p0(message: str) -> None:
    assert has_post_payment_refund_evidence(message) is False
    assert "refund" not in codes_from_text(message)
    assert classify_answer_safety(client_message=message).manager_only is False
    assert p0_pre_gate(message) is None
