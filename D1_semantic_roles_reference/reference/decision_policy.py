from __future__ import annotations

"""Политика ответа поверх смысловых ролей (референс-реализация).

Принцип проекта: «слои ВЫЧИТАЮТ, не переписывают». Этот модуль НЕ генерирует
текст. Он берёт роли (semantic_roles.MessageRoles) и текущее P0-решение и
возвращает ПЛАН: какие темы обязательно закрыть, куда маршрутизировать,
какие связки запрещены, и можно ли вообще ставить шаблон.

Закрывает 4 из 5 корневых пунктов round-4:
  (1) мультитема — вернуть СПИСОК тем к ответу, а не одну;
  (2) предпродажный возврат vs спор — типизированный refund_frame решает P0,
      а не растущий список regex;
  (3) запрет «рассрочка при маткапитале» — разные оси оплаты не смешиваем;
  (4) шаблон ТОЛЬКО как fallback — template_allowed=False, если есть
      содержательный ответ.

Пятый пункт (заземлённая генерация многотемного ответа) — это инструкция в
промпте генерации, не код; в ТЗ описан отдельно.
"""

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from semantic_roles import MessageRoles


DECISION_POLICY_SCHEMA_VERSION = "decision_policy_ref_v1_2026_05_25"


@dataclass(frozen=True)
class AnswerPlan:
    answer_topics: tuple[str, ...] = ()      # все темы, которые ОБЯЗАН закрыть ответ
    route: str = "bot_answer_self"           # bot_answer_self | manager_only | answer_then_manager
    p0_required: bool = False
    forbidden_pairs: tuple[str, ...] = ()    # связки, которые в одном ответе НЕЛЬЗЯ называть
    template_allowed: bool = False           # шаблон можно ставить ТОЛЬКО как fallback
    notes: tuple[str, ...] = ()
    evidence: Mapping[str, str] = field(default_factory=dict)

    def to_json_dict(self) -> Mapping[str, object]:
        return {
            "schema_version": DECISION_POLICY_SCHEMA_VERSION,
            "answer_topics": list(self.answer_topics),
            "route": self.route,
            "p0_required": self.p0_required,
            "forbidden_pairs": list(self.forbidden_pairs),
            "template_allowed": self.template_allowed,
            "notes": list(self.notes),
            "evidence": dict(self.evidence),
        }


# Темы, которые сами по себе ведут к менеджеру (P0), независимо от ролей.
_P0_TOPICS = {"refund_dispute"}


def build_answer_plan(
    roles: MessageRoles,
    *,
    external_p0: bool = False,
    substantive_answer_present: bool = False,
) -> AnswerPlan:
    """Собрать план ответа из ролей.

    external_p0 — P0 от прочих детекторов (legal/complaint/payment_dispute,
        p0_latch и т.п.). Этот модуль НЕ ослабляет внешний P0 — только добавляет
        свой (refund dispute) и решает предпродажный возврат.
    substantive_answer_present — есть ли уже содержательный прямой ответ/handoff.
        Если да → шаблон запрещён (только вычитание стиля, не перезапись).
    """

    notes: list[str] = []
    evidence: dict[str, str] = dict(roles.evidence)

    # --- (2) Возврат: тип решает P0 -------------------------------------- #
    refund_dispute = roles.refund_frame == "dispute"
    refund_presale = roles.refund_frame == "presale_policy"
    if refund_presale:
        notes.append("refund_presale_policy_is_benign_answer_not_p0")
    if refund_dispute:
        notes.append("refund_dispute_is_p0_manager")

    p0_required = bool(external_p0 or refund_dispute)

    # --- (1) Мультитема: собрать ВСЕ темы к закрытию ---------------------- #
    answer_topics = _ordered_answer_topics(roles, p0_required=p0_required)

    # --- (3) Запрет смешения осей оплаты --------------------------------- #
    forbidden_pairs: list[str] = []
    if roles.payment_source == "matkap":
        # маткапитал — ИСТОЧНИК оплаты; рассрочка/долями — СПОСОБ. В одном ответе
        # не предлагать рассрочку/долями вместе с маткапиталом (разные оси).
        forbidden_pairs.append("matkap+installment")
        notes.append("matkap_is_payment_source_do_not_offer_installment_in_same_answer")
        # убрать installment из тем ответа, если клиент сам про рассрочку не спросил
        if "installment" in answer_topics and roles.payment_method == "":
            answer_topics = tuple(t for t in answer_topics if t != "installment")

    # --- маршрут --------------------------------------------------------- #
    if p0_required:
        # если в одной реплике есть и спор-возврат, и безопасные темы — сперва P0
        route = "manager_only"
    else:
        route = "bot_answer_self"

    # --- (4) Шаблон только как fallback ---------------------------------- #
    template_allowed = not substantive_answer_present and not p0_required and not answer_topics
    if substantive_answer_present:
        notes.append("substantive_answer_present_template_must_only_refine_style")

    return AnswerPlan(
        answer_topics=answer_topics,
        route=route,
        p0_required=p0_required,
        forbidden_pairs=tuple(dict.fromkeys(forbidden_pairs)),
        template_allowed=template_allowed,
        notes=tuple(dict.fromkeys(notes)),
        evidence=evidence,
    )


def _ordered_answer_topics(roles: MessageRoles, *, p0_required: bool) -> tuple[str, ...]:
    """Темы к ответу в разумном порядке. P0-темы (спор-возврат) не попадают в
    список безопасного бот-ответа — их обрабатывает менеджер."""
    topics = [t for t in roles.topics if t not in _P0_TOPICS]
    # refund_presale остаётся: на предпродажный возврат бот безопасно отвечает
    # (политика возврата без личных обещаний). dispute уже исключён выше.
    return tuple(dict.fromkeys(topics))
