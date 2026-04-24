from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Optional


def _parse_datetime(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    candidate = str(value).strip()
    if not candidate:
        return None
    candidate = candidate.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _normalized(value: Any) -> str:
    return str(value or "").strip().casefold()


def _is_closed_stage(stage: Any) -> bool:
    normalized = _normalized(stage)
    if not normalized:
        return False
    closed_markers = (
        "closed",
        "lost",
        "won",
        "успеш",
        "неусп",
        "закры",
        "отказ",
        "архив",
    )
    return any(marker in normalized for marker in closed_markers)


@dataclass(frozen=True)
class RankedOpportunity:
    opportunity: dict[str, Any]
    score: float
    reasons: tuple[str, ...]


def rank_opportunity_candidates(
    *,
    call_started_at: datetime,
    opportunities: Iterable[dict[str, Any]],
    expected_branch: Optional[str] = None,
    expected_manager_id: Optional[str] = None,
    expected_manager_name: Optional[str] = None,
) -> list[RankedOpportunity]:
    ranked: list[RankedOpportunity] = []
    for opportunity in opportunities:
        score = 0.0
        reasons: list[str] = []

        sales_stage = opportunity.get("sales_stage")
        if _is_closed_stage(sales_stage):
            score += 0.15
            reasons.append("closed_stage")
        else:
            score += 0.55
            reasons.append("active_stage")

        if expected_branch and _normalized(opportunity.get("filial")) == _normalized(expected_branch):
            score += 0.1
            reasons.append("branch_match")

        if expected_manager_id and _normalized(opportunity.get("assigned_user_id")) == _normalized(expected_manager_id):
            score += 0.12
            reasons.append("manager_id_match")
        elif expected_manager_name and _normalized(opportunity.get("assigned_user_name")) == _normalized(expected_manager_name):
            score += 0.1
            reasons.append("manager_name_match")

        for field_name in ("date_modified", "system_date_closed", "date_closed", "date_entered"):
            dt_value = _parse_datetime(opportunity.get(field_name))
            if dt_value is None:
                continue
            delta_hours = abs((call_started_at - dt_value).total_seconds()) / 3600
            if delta_hours <= 24:
                score += 0.2
                reasons.append(f"{field_name}_within_24h")
            elif delta_hours <= 24 * 7:
                score += 0.12
                reasons.append(f"{field_name}_within_7d")
            elif delta_hours <= 24 * 30:
                score += 0.05
                reasons.append(f"{field_name}_within_30d")
            break

        ranked.append(
            RankedOpportunity(
                opportunity=opportunity,
                score=round(min(score, 0.99), 4),
                reasons=tuple(reasons),
            )
        )

    ranked.sort(key=lambda item: item.score, reverse=True)
    return ranked


def choose_best_opportunity(
    *,
    call_started_at: datetime,
    opportunities: Iterable[dict[str, Any]],
    expected_branch: Optional[str] = None,
    expected_manager_id: Optional[str] = None,
    expected_manager_name: Optional[str] = None,
    min_confidence: float = 0.45,
    ambiguity_gap: float = 0.05,
) -> tuple[Optional[dict[str, Any]], list[RankedOpportunity], bool]:
    ranked = rank_opportunity_candidates(
        call_started_at=call_started_at,
        opportunities=opportunities,
        expected_branch=expected_branch,
        expected_manager_id=expected_manager_id,
        expected_manager_name=expected_manager_name,
    )
    if not ranked:
        return None, [], False
    if ranked[0].score < min_confidence:
        return None, ranked, True
    if len(ranked) > 1 and abs(ranked[0].score - ranked[1].score) < ambiguity_gap:
        return None, ranked, True
    return ranked[0].opportunity, ranked, False


__all__ = [
    "RankedOpportunity",
    "choose_best_opportunity",
    "rank_opportunity_candidates",
]
