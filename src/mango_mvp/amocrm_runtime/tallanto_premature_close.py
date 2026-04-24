from __future__ import annotations

from dataclasses import dataclass


PREMATURE_CLOSE_RISK_VALUES = (
    "no_risk",
    "low",
    "medium",
    "high",
    "critical",
    "manual_review",
)

CLOSE_VERDICT_VALUES = (
    "closed_valid",
    "closed_too_early",
    "follow_up_needed",
    "reopen_recommended",
    "alternative_offer_needed",
    "manual_review",
)


@dataclass(frozen=True)
class PrematureCloseSignals:
    final_hard_rejection: bool = False
    has_follow_up_language: bool = False
    delayed_decision_language: bool = False
    alternative_offer_signal: bool = False
    recent_meaningful_call_count: int = 0
    has_active_group_membership: bool = False
    has_active_amo_work: bool = False
    has_payment_history: bool = False
    overdue_follow_up: bool = False
    ambiguous_identity: bool = False


@dataclass(frozen=True)
class PrematureCloseAssessment:
    risk: str
    verdict: str
    score: int
    reasons: tuple[str, ...]
    recommended_next_step: str


def assess_premature_close(signals: PrematureCloseSignals) -> PrematureCloseAssessment:
    if signals.ambiguous_identity:
        return PrematureCloseAssessment(
            risk="manual_review",
            verdict="manual_review",
            score=0,
            reasons=("ambiguous_identity",),
            recommended_next_step="Проверить вручную, к какому ученику и какой сделке относится кейс.",
        )

    score = 0
    reasons: list[str] = []

    if signals.final_hard_rejection:
        score -= 80
        reasons.append("hard_rejection")
    if signals.has_follow_up_language:
        score += 25
        reasons.append("follow_up_language")
    if signals.delayed_decision_language:
        score += 20
        reasons.append("delayed_decision_language")
    if signals.alternative_offer_signal:
        score += 25
        reasons.append("alternative_offer_signal")
    if signals.recent_meaningful_call_count >= 2:
        score += 20
        reasons.append("multiple_recent_calls")
    elif signals.recent_meaningful_call_count == 1:
        score += 10
        reasons.append("single_recent_call")
    if signals.has_payment_history:
        score += 10
        reasons.append("payment_history")
    if signals.overdue_follow_up:
        score += 20
        reasons.append("overdue_follow_up")
    if signals.has_active_group_membership:
        score -= 45
        reasons.append("already_in_active_group")
    if signals.has_active_amo_work:
        score -= 35
        reasons.append("active_amo_work")

    if score <= -40:
        return PrematureCloseAssessment(
            risk="no_risk",
            verdict="closed_valid",
            score=score,
            reasons=tuple(reasons),
            recommended_next_step="Не возвращать в активную продажу, если не появится новый сигнал интереса.",
        )
    if signals.alternative_offer_signal and score >= 25:
        return PrematureCloseAssessment(
            risk="medium" if score < 50 else "high",
            verdict="alternative_offer_needed",
            score=score,
            reasons=tuple(reasons),
            recommended_next_step="Подобрать альтернативный продукт или формат обучения и связаться повторно.",
        )
    if score >= 65:
        return PrematureCloseAssessment(
            risk="critical" if score >= 80 else "high",
            verdict="reopen_recommended",
            score=score,
            reasons=tuple(reasons),
            recommended_next_step="Вернуть сделку в работу и поставить срочный follow-up.",
        )
    if score >= 35:
        return PrematureCloseAssessment(
            risk="medium",
            verdict="follow_up_needed",
            score=score,
            reasons=tuple(reasons),
            recommended_next_step="Назначить follow-up и проверить, не была ли сделка закрыта слишком рано.",
        )
    return PrematureCloseAssessment(
        risk="low",
        verdict="closed_too_early" if score > 0 else "closed_valid",
        score=score,
        reasons=tuple(reasons),
        recommended_next_step=(
            "Сделать мягкое повторное касание, если это не создаёт лишнего давления."
            if score > 0
            else "Оставить в архиве, но сохранять в reactivation pool."
        ),
    )


__all__ = [
    "CLOSE_VERDICT_VALUES",
    "PREMATURE_CLOSE_RISK_VALUES",
    "PrematureCloseAssessment",
    "PrematureCloseSignals",
    "assess_premature_close",
]
