from __future__ import annotations

"""Answer plan built from typed semantic roles.

The plan does not generate customer text. It exposes required answer topics,
forbidden topic pairs and template permission so downstream layers subtract risk
instead of overwriting a precise answer with a generic fallback.
"""

from dataclasses import dataclass, field
from typing import Any, Mapping

from mango_mvp.channels.semantic_roles import MessageRoles


ANSWER_PLAN_SCHEMA_VERSION = "answer_plan_v1_2026_05_25"

P0_TOPICS = {"refund_dispute"}


@dataclass(frozen=True)
class AnswerPlan:
    answer_topics: tuple[str, ...] = ()
    route: str = "bot_answer_self"
    p0_required: bool = False
    forbidden_pairs: tuple[str, ...] = ()
    template_allowed: bool = False
    notes: tuple[str, ...] = ()
    evidence: Mapping[str, str] = field(default_factory=dict)

    def to_prompt_view(self) -> Mapping[str, Any]:
        return {
            "schema_version": ANSWER_PLAN_SCHEMA_VERSION,
            "answer_topics": list(self.answer_topics),
            "route": self.route,
            "p0_required": self.p0_required,
            "forbidden_pairs": list(self.forbidden_pairs),
            "template_allowed": self.template_allowed,
            "notes": list(self.notes),
            "evidence": dict(self.evidence),
        }


def build_answer_plan(
    roles: MessageRoles,
    *,
    external_p0: bool = False,
    substantive_answer_present: bool = False,
) -> AnswerPlan:
    notes: list[str] = []
    evidence: dict[str, str] = dict(roles.evidence)

    refund_dispute = roles.refund_frame == "dispute"
    refund_presale = roles.refund_frame == "presale_policy"
    if refund_presale:
        notes.append("refund_presale_policy_is_benign_answer_not_p0")
    if refund_dispute:
        notes.append("refund_dispute_is_p0_manager")

    p0_required = bool(external_p0 or refund_dispute)
    answer_topics = _ordered_answer_topics(roles)

    forbidden_pairs: list[str] = []
    if roles.payment_source == "matkap":
        forbidden_pairs.append("matkap+installment")
        notes.append("matkap_is_payment_source_do_not_offer_installment_in_same_answer")
        if "installment" in answer_topics and not roles.payment_method:
            answer_topics = tuple(topic for topic in answer_topics if topic != "installment")

    if substantive_answer_present:
        notes.append("substantive_answer_present_template_must_only_refine_style")

    return AnswerPlan(
        answer_topics=answer_topics,
        route="manager_only" if p0_required else "bot_answer_self",
        p0_required=p0_required,
        forbidden_pairs=tuple(dict.fromkeys(forbidden_pairs)),
        template_allowed=not substantive_answer_present and not p0_required and not answer_topics,
        notes=tuple(dict.fromkeys(notes)),
        evidence=evidence,
    )


def _ordered_answer_topics(roles: MessageRoles) -> tuple[str, ...]:
    return tuple(dict.fromkeys(topic for topic in roles.topics if topic not in P0_TOPICS))
