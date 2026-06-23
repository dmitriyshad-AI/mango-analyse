#!/usr/bin/env python3
"""Offline deal-progression re-judge for dynamic Telegram transcripts.

The LLM part is intentionally narrow: it returns only boolean observations for
one bot turn. Stage and verdict are computed deterministically in this file.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import run_telegram_dynamic_client_sim as sim


OBSERVATION_FIELDS: tuple[str, ...] = (
    "asked_missing_key",
    "named_concrete_option",
    "gave_conditions",
    "left_dated_followup",
    "requested_enrollment_data",
    "confirmed_slot",
    "sent_payment_path",
    "confirmed_access_or_docs",
    "pushed_sale",
    "handed_off_to_manager",
    "reasked_known",
    "dumped_catalog",
    "gave_fit_reason",
    "acknowledged_objection",
    "asked_constraint_question",
)
_OBSERVATION_ALIASES: Mapping[str, str] = {
    "named_concrete_offer": "named_concrete_option",
    "named_concrete_option_with_reason": "named_concrete_option",
    "pulled_constraint": "asked_constraint_question",
    "objection_to_checkable_step": "acknowledged_objection",
}
_FACT_RISK_LEVELS = frozenset({"no_match", "other_brand_match", "wrong_scope"})
_NUMBER_RISK_LEVELS = frozenset({"kb_integrity_issue", "no_match", "other_brand_match"})

STAGE_ORDER: Mapping[str, int] = {
    "S0": 0,
    "S1": 1,
    "S2": 2,
    "S3": 3,
    "S4": 4,
    "S5": 5,
    "S6": 6,
    "S7": 7,
    "S8": 8,
}
ORDER_STAGE = {value: key for key, value in STAGE_ORDER.items()}
VALID_STAGE_STARTS = set(STAGE_ORDER) | {"P0"}
STAGE_TARGETS = set(STAGE_ORDER) | {"hold", "route"}
TURN_VERDICTS = {"advanced", "held_ok", "stalled", "false_push", "mis_routed", "wrong_move"}


@dataclass(frozen=True)
class TurnAssessment:
    observation: Mapping[str, bool]
    stage_reached: str
    turn_verdict: str
    move_quality: str
    move_criteria_hit: tuple[str, ...]
    next_step: bool
    business_errors: tuple[str, ...]
    note: str


class ProgressionFakeJudgeModel:
    """Small deterministic model for smoke tests and no-LLM development."""

    def generate(self, prompt: str) -> Mapping[str, Any]:
        text = _extract_current_bot_text_from_prompt(prompt).casefold()
        observations = {field: False for field in OBSERVATION_FIELDS}
        if any(marker in text for marker in ("уточн", "какой класс", "предмет", "формат")):
            observations["asked_missing_key"] = True
        if any(marker in text for marker in ("какие дни", "удобнее", "цель", "уровень", "ограничен")):
            observations["asked_constraint_question"] = True
        if any(marker in text for marker in ("подойд", "вариант", "групп", "курс", "программа")):
            observations["named_concrete_option"] = True
        if any(marker in text for marker in ("потому", "поэтому", "так как", "вам подойд", "для вашей задачи")):
            observations["gave_fit_reason"] = True
        if any(marker in text for marker in ("стоим", "цена", "рассроч", "скидк", "услов")):
            observations["gave_conditions"] = True
        if any(marker in text for marker in ("понимаю", "вижу", "согласен", "да, это важно", "ваше опасение")):
            observations["acknowledged_objection"] = True
        if any(marker in text for marker in ("завтра", "сегодня", "дата", "перезвон", "напишем")):
            observations["left_dated_followup"] = True
        if any(marker in text for marker in ("фио", "почт", "email", "данные", "телефон")):
            observations["requested_enrollment_data"] = True
        if any(marker in text for marker in ("слот", "место", "запиш", "заброни")):
            observations["confirmed_slot"] = True
        if any(marker in text for marker in ("оплат", "счёт", "счет", "квитанц", "ссылк")):
            observations["sent_payment_path"] = True
        if any(marker in text for marker in ("доступ", "договор", "лиценз", "документ")):
            observations["confirmed_access_or_docs"] = True
        if any(marker in text for marker in ("покуп", "успей", "оформ", "продолжим обучение")):
            observations["pushed_sale"] = True
        if any(marker in text for marker in ("менеджер", "куратор", "ответственн", "бухгалтер")):
            observations["handed_off_to_manager"] = True
        if any(marker in text for marker in ("как я уже писал", "вы уже указали", "я же написал")):
            observations["reasked_known"] = True
        if any(marker in text for marker in ("у нас много", "есть разные", "можете выбрать любой", "все направления")):
            observations["dumped_catalog"] = True
        return observations


class JudgeCallCounter:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counts: Counter[str] = Counter()

    def increment(self, role: str) -> None:
        with self._lock:
            self._counts[str(role)] += 1

    def snapshot(self) -> Mapping[str, int]:
        with self._lock:
            return dict(self._counts)


class CountingJudgeModel:
    def __init__(self, inner: Any, *, counter: JudgeCallCounter, role: str) -> None:
        self.inner = inner
        self.counter = counter
        self.role = role

    def generate(self, prompt: str) -> Mapping[str, Any]:
        self.counter.increment(self.role)
        return self.inner.generate(prompt)


def _extract_current_bot_text_from_prompt(prompt: str) -> str:
    marker = "Текущий ответ бота:\n"
    start = prompt.find(marker)
    if start < 0:
        return prompt
    start += len(marker)
    end = prompt.find("\n\nТранскрипт", start)
    if end < 0:
        return prompt[start:]
    return prompt[start:end]


_HANDOFF_TEXT_RE = re.compile(
    r"("
    r"\b(?:передам|передаю|передадим|направлю|направим)\b.{0,90}\b(?:менеджер\w*|куратор\w*|бухгалтер\w*|ответственн\w*)|"
    r"\b(?:менеджер|куратор|бухгалтер|ответственн\w*)\b.{0,90}\b(?:уточн\w*|провер\w*|подготов\w*|свяж\w*|напиш\w*|ответ\w*|верн[её]т\w*|сообщ\w*)|"
    r"\bне\s+могу\s+(?:подсказать|проверить|подтвердить|сориентировать)\b|"
    r"\b(?:нужно|надо)\s+уточнить\s+у\s+(?:менеджер\w*|куратор\w*|бухгалтер\w*)|"
    r"\bзапрос\b.{0,90}\bпередать\s+в\s+бухгалтер\w*"
    r")",
    re.IGNORECASE | re.DOTALL,
)

_SUBSTANTIVE_ANSWER_RE = re.compile(
    r"("
    r"\b\d[\d\s]*(?:₽|руб\.?|р\b)|"
    r"\b\d{1,2}[:.]\d{2}\b|"
    r"\b(?:стоим\w*|цен[аы]|расписан\w*|адрес\w*|рассроч\w*|скидк\w*|стоить)\b|"
    r"\b(?:после\s+подтверждени[яе]\s+оплат\w*|после\s+оплат\w*|на\s+почту\s+прид[её]т|приглашени[ея]\s+на\s+учебн\w+\s+платформ\w*)\b|"
    r"\b(?:soholms|mts[-\s]?link|учебн\w+\s+платформ\w*|ссылк\w+\s+и\s+инструкци\w*)\b|"
    r"\b(?:вступительн\w+\s+тест\w*|заполня\w+\s+анкет\w*|анкет\w+\s+и\s+вступительн\w+\s+тест\w*|распределени\w+\s+по\s+групп\w*)\b|"
    r"\b(?:есть|доступн\w*)\s+(?:базов\w+|продвинут\w+|очная|очный|онлайн|групп\w+|курс\w+|программ\w+)"
    r")",
    re.IGNORECASE | re.DOTALL,
)


def _text_hands_off_to_manager(bot_text: str) -> bool:
    text = str(bot_text or "").strip()
    if not text or not _HANDOFF_TEXT_RE.search(text):
        return False
    return not bool(_SUBSTANTIVE_ANSWER_RE.search(text))


def _has_substantive_observation(observation: Mapping[str, bool]) -> bool:
    return any(
        bool(observation.get(field))
        for field in (
            "named_concrete_option",
            "gave_conditions",
            "left_dated_followup",
            "requested_enrollment_data",
            "confirmed_slot",
            "sent_payment_path",
            "confirmed_access_or_docs",
            "pushed_sale",
        )
    )


def normalize_observations(payload: Mapping[str, Any], *, bot_text: str = "") -> dict[str, bool]:
    observations = {field: bool(payload.get(field)) for field in OBSERVATION_FIELDS}
    for source, target in _OBSERVATION_ALIASES.items():
        if bool(payload.get(source)):
            observations[target] = True
    observations["handed_off_to_manager"] = _text_hands_off_to_manager(bot_text) and not _has_substantive_observation(
        observations
    )
    return observations


def build_progression_prompt(
    *,
    persona: Mapping[str, Any],
    turns: Sequence[Mapping[str, Any]],
    current_turn_index: int,
) -> str:
    transcript_lines: list[str] = []
    for turn in turns[: current_turn_index + 1]:
        turn_no = int(turn.get("turn") or len(transcript_lines) + 1)
        transcript_lines.append(
            f"Ход {turn_no}\n"
            f"Клиент: {turn.get('client_message') or ''}\n"
            f"Бот: {turn.get('bot_text') or ''}"
        )
    current_turn = turns[current_turn_index]
    safe_persona = {
        "dialog_id": persona.get("dialog_id"),
        "brand": persona.get("brand"),
        "held_facts": persona.get("held_facts") or {},
        "deal_state": (persona.get("progression_tags") or {}).get("deal_state") or {},
        "stage_start": (persona.get("progression_tags") or {}).get("stage_start"),
    }
    return (
        "Ты узкий судья наблюдений по одному ответу Telegram-бота образовательного центра.\n"
        "Верни ТОЛЬКО JSON-объект с булевыми полями ниже. Не возвращай стадию, verdict, оценку качества, "
        "советы или текстовые объяснения.\n\n"
        "Важно: stage_target в персоне, если он где-то встречался, является ожиданием теста, а не фактом. "
        "Не подгоняй наблюдения под ожидание. Отмечай только то, что явно сделал бот в текущем ответе.\n\n"
        "Поля JSON:\n"
        "- asked_missing_key: бот задал один недостающий ключевой вопрос: класс, предмет, формат, цель, город.\n"
        "- asked_constraint_question: бот задал уточняющий вопрос об ограничении/предпочтении: дни, цель, уровень, формат, бюджет.\n"
        "- named_concrete_option: бот назвал конкретный продукт, группу, уровень, вариант или слот, а не общую рекламу.\n"
        "- gave_fit_reason: бот объяснил, почему именно этот вариант подходит клиенту.\n"
        "- gave_conditions: бот дал условия: цену, скидку, рассрочку, время, формат, ограничение или снял возражение.\n"
        "- acknowledged_objection: бот признал возражение/опасение клиента перед ответом или проверяемым шагом.\n"
        "- left_dated_followup: бот оставил материал/перезвон/следующий контакт с датой, сроком или явным условием.\n"
        "- requested_enrollment_data: бот запросил данные для записи: ФИО, класс, почту, телефон, направление.\n"
        "- confirmed_slot: бот подтвердил или предложил запись, место, слот, группу, бронь.\n"
        "- sent_payment_path: бот дал или предложил счёт, квитанцию, ссылку, QR, способ оплаты.\n"
        "- confirmed_access_or_docs: бот объяснил доступ после оплаты, документы, договор, лицензию или сервисный шаг.\n"
        "- pushed_sale: бот дожимал продажу, оформление, оплату или новый продукт.\n"
        "- handed_off_to_manager: бот реально пасует задачу менеджеру, куратору, бухгалтерии или ответственному "
        "сотруднику, а не просто упоминает сотрудника рядом с содержательным ответом.\n\n"
        "- reasked_known: бот переспросил класс, предмет, формат, оплату или другой ключ, который клиент уже сообщил выше.\n"
        "- dumped_catalog: бот вывалил общий каталог/много вариантов без выбора или рекомендации под клиента.\n\n"
        "Оценивай текущий ход бота, а не всю историю. Синонимы засчитывай. Если признаки нет явно — false.\n\n"
        f"Персона без целевого якоря:\n{json.dumps(safe_persona, ensure_ascii=False, indent=2)}\n\n"
        f"Текущий ход для оценки: {current_turn.get('turn')}\n"
        f"Текущий ответ бота:\n{current_turn.get('bot_text') or ''}\n\n"
        "Транскрипт до текущего хода включительно:\n"
        f"{chr(10).join(transcript_lines)}\n"
    )


def stage_from_observations(observation: Mapping[str, bool], *, start_stage: str) -> str:
    reached = STAGE_ORDER.get(start_stage, 0)
    if observation.get("asked_missing_key") or observation.get("asked_constraint_question"):
        reached = max(reached, 1)
    if observation.get("named_concrete_option"):
        reached = max(reached, 2)
    if observation.get("gave_conditions"):
        reached = max(reached, 3)
    if observation.get("left_dated_followup"):
        reached = max(reached, 4)
    if observation.get("requested_enrollment_data") or observation.get("confirmed_slot"):
        reached = max(reached, 5)
    if observation.get("sent_payment_path"):
        reached = max(reached, 6)
    if observation.get("confirmed_access_or_docs"):
        reached = max(reached, 8)
    return ORDER_STAGE[reached]


def has_valid_next_step(observation: Mapping[str, bool], *, stage_target: str, p0_turn: bool) -> bool:
    if p0_turn:
        return bool(observation.get("handed_off_to_manager")) and not _collects_p0_details(observation)
    if stage_target == "route":
        return bool(observation.get("handed_off_to_manager"))
    return any(
        bool(observation.get(field))
        for field in (
            "asked_missing_key",
            "asked_constraint_question",
            "named_concrete_option",
            "left_dated_followup",
            "requested_enrollment_data",
            "confirmed_slot",
            "sent_payment_path",
            "confirmed_access_or_docs",
        )
    )


def assess_move_quality(
    observation: Mapping[str, bool],
    *,
    persona: Mapping[str, Any],
    turn: Mapping[str, Any],
    stage_start: str,
    stage_target: str,
    p0_turn: bool,
    route_handoff_already_done: bool = False,
    audit_errors: Sequence[str] = (),
) -> tuple[str, tuple[str, ...]]:
    """Assess how well the current move was made, independently of reached stage."""

    if audit_errors:
        return "wrong", ("fabrication_blocked",)
    if p0_turn:
        if observation.get("handed_off_to_manager") and not _collects_p0_details(observation):
            return "winning", ("p0_handoff",)
        return "wrong", ("p0_mishandled",)
    if stage_target == "route":
        if observation.get("handed_off_to_manager"):
            return "winning", ("manager_route",)
        if route_handoff_already_done and _route_followup_after_handoff(observation):
            return "weak", ("post_handoff_followup",)
        return "wrong", ("missed_manager_route",)

    paid_context = _paid_or_service_context(persona, stage_start=stage_start)
    if paid_context:
        if observation.get("pushed_sale"):
            return "wrong", ("redrive_after_pay",)
        if _service_not_resell(observation, persona=persona, stage_start=stage_start):
            return "winning", ("service_not_resell",)
        if observation.get("handed_off_to_manager") or observation.get("left_dated_followup"):
            return "weak", ("service_followup",)

    if _sibling_trap(persona):
        if observation.get("sent_payment_path") or observation.get("confirmed_access_or_docs") or observation.get("pushed_sale"):
            return "wrong", ("sibling_stage_carried",)
        if observation.get("asked_missing_key") or observation.get("asked_constraint_question"):
            return "winning", ("sibling_new_need_requalified",)
        return "weak", ("sibling_not_requalified",)

    if observation.get("handed_off_to_manager"):
        return "wrong", ("unneeded_manager_handoff",)
    if stage_target == "hold":
        if observation.get("pushed_sale") and not _close_or_thanks(turn):
            return "wrong", ("false_push_after_close",)
        return "winning", ("held_without_resell",)
    if observation.get("reasked_known"):
        return "weak", ("reasked_known",)
    if observation.get("dumped_catalog"):
        return "weak", ("dumped_catalog",)

    if stage_target == "S2":
        if observation.get("named_concrete_option") and observation.get("gave_fit_reason"):
            return "winning", ("concrete_option_with_fit_reason",)
        if observation.get("named_concrete_option") or observation.get("asked_missing_key") or observation.get(
            "asked_constraint_question"
        ):
            return "weak", ("partial_qualification_or_offer",)
    if stage_target == "S3":
        if observation.get("gave_conditions") and observation.get("acknowledged_objection"):
            return "winning", ("objection_acknowledged_with_conditions",)
        if observation.get("gave_conditions"):
            return "winning", ("conditions_answered",)
        if observation.get("acknowledged_objection"):
            return "weak", ("objection_acknowledged_without_conditions",)
    if stage_target == "S5":
        if observation.get("requested_enrollment_data") or observation.get("confirmed_slot"):
            return "winning", ("enrollment_step_requested",)
        if observation.get("named_concrete_option") or observation.get("gave_fit_reason"):
            return "weak", ("option_without_enrollment_step",)
    if stage_target == "S6":
        if observation.get("sent_payment_path"):
            return "winning", ("payment_path_sent",)
        if observation.get("requested_enrollment_data") or observation.get("confirmed_slot"):
            return "weak", ("prepayment_step_only",)
    if stage_target == "S8":
        if observation.get("confirmed_access_or_docs"):
            return "winning", ("access_or_docs_confirmed",)

    if observation.get("asked_missing_key") or observation.get("asked_constraint_question"):
        return "weak", ("asked_next_question",)
    if observation.get("named_concrete_option") or observation.get("gave_conditions") or observation.get("left_dated_followup"):
        return "weak", ("substantive_but_incomplete",)
    return "wrong", ("no_sales_move",)


def _turn_fact_audit_errors(turn: Mapping[str, Any]) -> tuple[str, ...]:
    errors: list[str] = []
    number_audit = turn.get("number_audit") if isinstance(turn.get("number_audit"), Mapping) else {}
    if bool(number_audit.get("has_risky_number")) or _audit_has_levels(number_audit, _NUMBER_RISK_LEVELS):
        errors.append("fabrication_in_move")

    fact_audit = turn.get("judge_fact_audit") if isinstance(turn.get("judge_fact_audit"), Mapping) else {}
    fact_levels = _audit_levels(fact_audit)
    if fact_levels & _FACT_RISK_LEVELS:
        errors.append("fabrication_in_move")
    if "wrong_scope" in fact_levels:
        errors.append("wrong_venue_or_fact")
    return tuple(dict.fromkeys(errors))


def _audit_has_levels(audit: Mapping[str, Any], levels: set[str] | frozenset[str]) -> bool:
    return bool(_audit_levels(audit) & set(levels))


def _audit_levels(audit: Mapping[str, Any]) -> set[str]:
    result: set[str] = set()
    for item in audit.get("items") or []:
        if isinstance(item, Mapping):
            level = str(item.get("level") or "").strip()
            if level:
                result.add(level)
    return result


def _service_not_resell(observation: Mapping[str, bool], *, persona: Mapping[str, Any], stage_start: str) -> bool:
    return (
        _paid_or_service_context(persona, stage_start=stage_start)
        and bool(observation.get("confirmed_access_or_docs"))
        and not bool(observation.get("pushed_sale"))
    )


def assess_turn(
    observation: Mapping[str, bool],
    *,
    persona: Mapping[str, Any],
    turn: Mapping[str, Any],
    stage_start: str,
    stage_target: str,
    route_handoff_already_done: bool = False,
) -> TurnAssessment:
    stage_reached = stage_from_observations(observation, start_stage=stage_start)
    p0_turn = _is_p0_turn(persona, turn, stage_start=stage_start, stage_target=stage_target)
    next_step = has_valid_next_step(observation, stage_target=stage_target, p0_turn=p0_turn)
    errors: list[str] = []
    audit_errors = _turn_fact_audit_errors(turn)
    move_quality, move_criteria_hit = assess_move_quality(
        observation,
        persona=persona,
        turn=turn,
        stage_start=stage_start,
        stage_target=stage_target,
        p0_turn=p0_turn,
        route_handoff_already_done=route_handoff_already_done,
        audit_errors=audit_errors,
    )

    if audit_errors:
        return TurnAssessment(
            observation,
            stage_reached,
            "wrong_move",
            move_quality,
            move_criteria_hit,
            False,
            tuple(audit_errors),
            "Факт-аудит/аудит чисел нашёл неподтверждённый или неверно scoped факт в ходе.",
        )

    if p0_turn:
        if observation.get("handed_off_to_manager") and not _collects_p0_details(observation):
            return TurnAssessment(
                observation,
                stage_reached,
                "held_ok",
                move_quality,
                move_criteria_hit,
                False,
                (),
                "P0: сразу передал ответственному без сбора данных.",
            )
        if not observation.get("handed_off_to_manager"):
            errors.append("under_handoff_p0")
        if _collects_p0_details(observation):
            errors.append("p0_mishandled_collect_first")
        return TurnAssessment(
            observation,
            stage_reached,
            "mis_routed",
            move_quality,
            move_criteria_hit,
            False,
            tuple(errors),
            "P0 обработан не как немедленный хендофф.",
        )

    if stage_target == "route":
        if observation.get("handed_off_to_manager"):
            return TurnAssessment(
                observation,
                stage_reached,
                "held_ok",
                move_quality,
                move_criteria_hit,
                True,
                (),
                "Правомерно передал менеджеру/ответственному.",
            )
        if route_handoff_already_done and _route_followup_after_handoff(observation):
            return TurnAssessment(
                observation,
                stage_reached,
                "held_ok",
                move_quality,
                move_criteria_hit,
                True,
                (),
                "После уже выполненной передачи удержал сервисную нить без самостоятельного решения.",
            )
        errors.append("under_handoff_service")
        return TurnAssessment(
            observation,
            stage_reached,
            "mis_routed",
            move_quality,
            move_criteria_hit,
            next_step,
            tuple(errors),
            "Нужна передача менеджеру, но бот ответил сам.",
        )

    paid_context = _paid_or_service_context(persona, stage_start=stage_start)
    if paid_context and observation.get("pushed_sale"):
        errors.append("redrive_after_pay")
        return TurnAssessment(
            observation,
            stage_reached,
            "false_push",
            move_quality,
            move_criteria_hit,
            next_step,
            tuple(errors),
            "После оплаты/сервиса бот снова дожимал продажу.",
        )

    if _sibling_trap(persona) and (
        observation.get("sent_payment_path") or observation.get("confirmed_access_or_docs") or observation.get("pushed_sale")
    ):
        errors.append("stage_carried_to_sibling")
        return TurnAssessment(
            observation,
            stage_reached,
            "false_push",
            move_quality,
            move_criteria_hit,
            next_step,
            tuple(errors),
            "Стадия старшего ребёнка перенесена на новую сделку.",
        )

    if observation.get("handed_off_to_manager"):
        errors.append("over_handoff_service")
        return TurnAssessment(
            observation,
            stage_reached,
            "mis_routed",
            move_quality,
            move_criteria_hit,
            next_step,
            tuple(errors),
            "Бот увёл к менеджеру то, что должен был вести сам.",
        )

    if stage_target == "hold":
        if observation.get("pushed_sale") and not _close_or_thanks(turn):
            errors.append("false_push_after_close")
            return TurnAssessment(
                observation,
                stage_reached,
                "false_push",
                move_quality,
                move_criteria_hit,
                next_step,
                tuple(errors),
                "Бот дожимал, хотя ожидалось удержание/закрытие.",
            )
        return TurnAssessment(
            observation,
            stage_reached,
            "held_ok",
            move_quality,
            move_criteria_hit,
            next_step,
            (),
            "Правомерно удержал диалог без лишнего дожима.",
        )

    target_order = STAGE_ORDER.get(stage_target)
    reached_order = STAGE_ORDER.get(stage_reached, STAGE_ORDER.get(stage_start, 0))
    if target_order is not None and reached_order >= target_order and (next_step or observation.get("gave_conditions")):
        return TurnAssessment(
            observation,
            stage_reached,
            "advanced",
            move_quality,
            move_criteria_hit,
            next_step,
            (),
            "Достиг или превысил целевую стадию по наблюдаемому действию.",
        )

    if not next_step:
        errors.append("no_valid_next_step")
    if target_order is not None and reached_order < target_order:
        errors.append("target_not_reached")
    return TurnAssessment(
        observation,
        stage_reached,
        "stalled",
        move_quality,
        move_criteria_hit,
        next_step,
        tuple(dict.fromkeys(errors)),
        "Содержательный ход не довёл до целевой стадии.",
    )


def assess_dialog(
    *,
    judge_model: Any | None,
    dialog: Mapping[str, Any],
    persona_by_id: Mapping[str, Mapping[str, Any]],
    turn_observations: Sequence[Mapping[str, Any]] | None = None,
) -> Mapping[str, Any]:
    dialog_id = str(dialog.get("dialog_id") or "")
    persona = dict(dialog.get("persona") or persona_by_id.get(dialog_id) or {})
    progression = dict(persona.get("progression_tags") or {})
    stage_start = _normalize_stage(str(progression.get("stage_start") or "S0"), fallback="S0")
    stage_target = _normalize_target(str(progression.get("stage_target") or stage_start), fallback=stage_start)
    turns = [turn for turn in (dialog.get("turns") or []) if isinstance(turn, Mapping)]

    assessments: list[TurnAssessment] = []
    route_handoff_seen = False
    for index, turn in enumerate(turns):
        if turn_observations is not None and index < len(turn_observations):
            raw = turn_observations[index]
        else:
            if judge_model is None:
                raise ValueError(f"Missing stored observation for dialog_id={dialog_id!r} turn_index={index}")
            prompt = build_progression_prompt(persona=persona, turns=turns, current_turn_index=index)
            raw = judge_model.generate(prompt)
        observation = normalize_observations(raw, bot_text=str(turn.get("bot_text") or ""))
        assessments.append(
            assess_turn(
                observation,
                persona=persona,
                turn=turn,
                stage_start=stage_start,
                stage_target=stage_target,
                route_handoff_already_done=route_handoff_seen,
            )
        )
        if stage_target == "route" and assessments[-1].observation.get("handed_off_to_manager"):
            route_handoff_seen = True

    if assessments:
        stage_reached = max(
            (assessment.stage_reached for assessment in assessments),
            key=lambda stage: STAGE_ORDER.get(stage, 0),
        )
    else:
        stage_reached = stage_start
    turn_verdicts = [item.turn_verdict for item in assessments]
    turn_move_quality = [item.move_quality for item in assessments]
    move_criteria_hit = [list(item.move_criteria_hit) for item in assessments]
    winning_move_rate = _ratio(turn_move_quality.count("winning"), len(turn_move_quality))
    next_steps = [item.next_step for item in assessments]
    business_errors = _dialog_business_errors(
        stage_reached=stage_reached,
        stage_target=stage_target,
        assessments=assessments,
    )
    dialog_verdict = _dialog_verdict(turn_verdicts, stage_target=stage_target)
    note = _dialog_note(assessments)

    return {
        "dialog_id": dialog_id,
        "brand": str(dialog.get("brand") or persona.get("brand") or ""),
        "stage_start": stage_start,
        "stage_target": stage_target,
        "stage_reached": stage_reached,
        "turn_verdicts": turn_verdicts,
        "turn_move_quality": turn_move_quality,
        "move_criteria_hit": move_criteria_hit,
        "winning_move_rate": winning_move_rate,
        "next_step_each_turn": next_steps,
        "dialog_verdict": dialog_verdict,
        "business_errors": business_errors,
        "note": note,
        "turn_observations": [dict(item.observation) for item in assessments],
    }


def _dialog_verdict(turn_verdicts: Sequence[str], *, stage_target: str) -> str:
    if any(item == "false_push" for item in turn_verdicts):
        return "false_push"
    if any(item == "wrong_move" for item in turn_verdicts):
        return "wrong_move"
    if any(item == "mis_routed" for item in turn_verdicts):
        return "mis_routed"
    if any(item == "advanced" for item in turn_verdicts):
        return "advanced"
    if any(item == "held_ok" for item in turn_verdicts):
        return "held_ok"
    if stage_target in {"hold", "route"} and not turn_verdicts:
        return "held_ok"
    return "stalled"


def _dialog_business_errors(
    *,
    stage_reached: str,
    stage_target: str,
    assessments: Sequence[TurnAssessment],
) -> list[str]:
    errors = list(dict.fromkeys(error for item in assessments for error in item.business_errors))
    target_order = STAGE_ORDER.get(stage_target)
    if target_order is not None and STAGE_ORDER.get(stage_reached, 0) >= target_order:
        errors = [error for error in errors if error != "target_not_reached"]
    return errors


def summarize_results(rows: Sequence[Mapping[str, Any]], *, llm_calls: Mapping[str, int] | None = None) -> Mapping[str, Any]:
    verdicts = Counter(str(row.get("dialog_verdict") or "") for row in rows)
    move_quality = Counter(
        str(item)
        for row in rows
        for item in (row.get("turn_move_quality") or [])
        if str(item)
    )
    dialogs = len(rows)
    turn_total = sum(len(row.get("next_step_each_turn") or []) for row in rows)
    next_step_total = sum(sum(1 for item in row.get("next_step_each_turn") or [] if item) for row in rows)
    good = verdicts.get("advanced", 0) + verdicts.get("held_ok", 0)
    winning_turns = move_quality.get("winning", 0)
    advanced_or_held_weak = 0
    for row in rows:
        qualities = [str(item) for item in (row.get("turn_move_quality") or [])]
        if row.get("dialog_verdict") in {"advanced", "held_ok"} and _ratio(qualities.count("winning"), len(qualities)) < 0.5:
            advanced_or_held_weak += 1
    call_counts = {str(key): int(value or 0) for key, value in (llm_calls or {}).items()}
    llm_summary = {
        "total": sum(call_counts.values()),
        "progression_judge": call_counts.get("progression_judge", 0),
        "progression_judge_fake": call_counts.get("progression_judge_fake", 0),
        "dialogs": dialogs,
        "turns": turn_total,
    }
    return {
        "dialogs": dialogs,
        "dialog_verdicts": dict(sorted(verdicts.items())),
        "advanced_or_held_ok_rate": _ratio(good, dialogs),
        "stalled_rate": _ratio(verdicts.get("stalled", 0), dialogs),
        "false_push_or_mis_routed_or_wrong_rate": _ratio(
            verdicts.get("false_push", 0) + verdicts.get("mis_routed", 0) + verdicts.get("wrong_move", 0),
            dialogs,
        ),
        "false_push_or_mis_routed_rate": _ratio(verdicts.get("false_push", 0) + verdicts.get("mis_routed", 0), dialogs),
        "valid_next_step_turn_rate": _ratio(next_step_total, turn_total),
        "winning_move_rate": _ratio(winning_turns, turn_total),
        "turn_move_quality": dict(sorted(move_quality.items())),
        "advanced_or_held_with_winning_move_rate_lt_0_5": advanced_or_held_weak,
        "turns": turn_total,
        "turns_with_valid_next_step": next_step_total,
        "business_errors": dict(sorted(Counter(error for row in rows for error in row.get("business_errors") or []).items())),
        "llm_calls": llm_summary,
    }


def render_summary_md(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Progression Judge Summary",
        "",
        f"- dialogs: {summary.get('dialogs')}",
        f"- turns: {summary.get('turns')}",
        f"- advanced+held_ok: {summary.get('advanced_or_held_ok_rate')}",
        f"- stalled: {summary.get('stalled_rate')}",
        f"- false_push+mis_routed+wrong: {summary.get('false_push_or_mis_routed_or_wrong_rate')}",
        f"- valid_next_step_turn_rate: {summary.get('valid_next_step_turn_rate')}",
        f"- winning_move_rate: {summary.get('winning_move_rate')}",
        f"- advanced/held with winning_move_rate<0.5: {summary.get('advanced_or_held_with_winning_move_rate_lt_0_5')}",
        f"- llm_calls: {summary.get('llm_calls')}",
        "",
        "## Verdicts",
    ]
    for verdict, count in (summary.get("dialog_verdicts") or {}).items():
        lines.append(f"- {verdict}: {count}")
    lines.extend(["", "## Business Errors"])
    for error, count in (summary.get("business_errors") or {}).items():
        lines.append(f"- {error}: {count}")
    lines.extend(["", "## Move Quality"])
    for quality, count in (summary.get("turn_move_quality") or {}).items():
        lines.append(f"- {quality}: {count}")
    return "\n".join(lines) + "\n"


def load_observations_by_dialog(path: Path | None) -> Mapping[str, Sequence[Mapping[str, Any]]]:
    if path is None:
        return {}
    rows = sim.load_transcripts(path)
    result: dict[str, Sequence[Mapping[str, Any]]] = {}
    for row in rows:
        dialog_id = str(row.get("dialog_id") or "").strip()
        observations = row.get("turn_observations")
        if dialog_id and isinstance(observations, Sequence) and not isinstance(observations, (str, bytes, bytearray)):
            result[dialog_id] = [item for item in observations if isinstance(item, Mapping)]
    return result


def build_judge_model(args: argparse.Namespace) -> Any:
    counter = getattr(args, "progression_llm_call_counter", None)
    if args.judge_mode == "fake":
        model: Any = ProgressionFakeJudgeModel()
        role = "progression_judge_fake"
    else:
        model = sim.CodexJsonModel(
            model=args.model,
            reasoning_effort=args.judge_reasoning,
            timeout_sec=args.timeout_sec,
            isolated=bool(args.codex_isolated),
        )
        role = "progression_judge"
    if isinstance(counter, JudgeCallCounter):
        return CountingJudgeModel(model, counter=counter, role=role)
    return model


def assess_dialogs(
    *,
    args: argparse.Namespace,
    dialogs: Sequence[Mapping[str, Any]],
    persona_by_id: Mapping[str, Mapping[str, Any]],
    observations_by_dialog: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
) -> list[Mapping[str, Any]]:
    parallel = max(1, int(getattr(args, "parallel", 1) or 1))
    if parallel == 1:
        judge_model = None if observations_by_dialog else build_judge_model(args)
        return [
            assess_dialog(
                judge_model=judge_model,
                dialog=dialog,
                persona_by_id=persona_by_id,
                turn_observations=(observations_by_dialog or {}).get(str(dialog.get("dialog_id") or "")),
            )
            for dialog in dialogs
        ]

    indexed_rows: dict[int, Mapping[str, Any]] = {}

    def _worker(index: int, dialog: Mapping[str, Any]) -> tuple[int, Mapping[str, Any]]:
        judge_model = None if observations_by_dialog else build_judge_model(args)
        return index, assess_dialog(
            judge_model=judge_model,
            dialog=dialog,
            persona_by_id=persona_by_id,
            turn_observations=(observations_by_dialog or {}).get(str(dialog.get("dialog_id") or "")),
        )

    with ThreadPoolExecutor(max_workers=parallel) as pool:
        futures = [pool.submit(_worker, index, dialog) for index, dialog in enumerate(dialogs)]
        for future in as_completed(futures):
            index, row = future.result()
            indexed_rows[index] = row
    return [indexed_rows[index] for index in range(len(dialogs))]


def _normalize_stage(value: str, *, fallback: str) -> str:
    text = str(value or "").strip().upper()
    return text if text in VALID_STAGE_STARTS else fallback


def _normalize_target(value: str, *, fallback: str) -> str:
    text = str(value or "").strip()
    upper = text.upper()
    if upper in STAGE_ORDER:
        return upper
    lowered = text.casefold()
    return lowered if lowered in STAGE_TARGETS else fallback


def _collects_p0_details(observation: Mapping[str, bool]) -> bool:
    return any(
        bool(observation.get(field))
        for field in (
            "asked_missing_key",
            "named_concrete_offer",
            "gave_conditions",
            "requested_enrollment_data",
            "confirmed_slot",
            "sent_payment_path",
            "pushed_sale",
        )
    )


def _route_followup_after_handoff(observation: Mapping[str, bool]) -> bool:
    return not any(
        bool(observation.get(field))
        for field in (
            "named_concrete_offer",
            "gave_conditions",
            "confirmed_slot",
            "sent_payment_path",
            "confirmed_access_or_docs",
            "pushed_sale",
        )
    )


def _is_p0_turn(persona: Mapping[str, Any], turn: Mapping[str, Any], *, stage_start: str, stage_target: str) -> bool:
    progression = persona.get("progression_tags") if isinstance(persona.get("progression_tags"), Mapping) else {}
    deal_state = progression.get("deal_state") if isinstance(progression.get("deal_state"), Mapping) else {}
    if stage_start == "P0" or str(stage_target).casefold() == "route" and deal_state.get("p0"):
        return True
    if str(persona.get("injected_p0") or "").strip():
        return True
    client_text = str(turn.get("client_message") or "").casefold()
    return bool(
        re.search(
            r"(возврат\w*|верните\s+деньги|жалоб\w*|претензи\w*|\bсуд(?:ом|а|е|у)?\b|юрист\w*|"
            r"не\s+приш[её]л\s+доступ|нет\s+доступа|оплат\w*[^.?!]{0,80}(?:\bнет\b|не\s+дали|не\s+приш))",
            client_text,
        )
    )


def _paid_or_service_context(persona: Mapping[str, Any], *, stage_start: str) -> bool:
    progression = persona.get("progression_tags") if isinstance(persona.get("progression_tags"), Mapping) else {}
    deal_state = progression.get("deal_state") if isinstance(progression.get("deal_state"), Mapping) else {}
    return stage_start in {"S7", "S8"} or bool(deal_state.get("paid") or deal_state.get("paid_claimed") or deal_state.get("existing_customer"))


def _sibling_trap(persona: Mapping[str, Any]) -> bool:
    progression = persona.get("progression_tags") if isinstance(persona.get("progression_tags"), Mapping) else {}
    deal_state = progression.get("deal_state") if isinstance(progression.get("deal_state"), Mapping) else {}
    traps = {str(item) for item in (progression.get("traps") or [])}
    return bool(deal_state.get("elder_paid_claimed") or "sibling_stage_carry" in traps)


def _close_or_thanks(turn: Mapping[str, Any]) -> bool:
    text = str(turn.get("client_message") or "").casefold()
    return bool(re.search(r"(спасибо|понял|поняла|благодар)", text))


def _dialog_note(assessments: Sequence[TurnAssessment]) -> str:
    for item in assessments:
        if item.business_errors:
            return item.note
    for item in reversed(assessments):
        if item.turn_verdict in {"advanced", "held_ok"}:
            return item.note
    return assessments[-1].note if assessments else "Нет ходов для оценки."


def _ratio(part: int, whole: int) -> float:
    if whole <= 0:
        return 0.0
    return round(part / whole, 4)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Re-score dynamic Telegram transcripts for deal progression.")
    parser.add_argument("--transcripts", type=Path, required=True, help="Existing dynamic_dialog_transcripts.jsonl.")
    parser.add_argument("--scenarios", type=Path, required=True, help="Scenario seed with progression_tags.")
    parser.add_argument("--out", type=Path, default=None, help="Output progression_results.jsonl.")
    parser.add_argument("--summary-out", type=Path, default=None, help="Output progression_summary.json.")
    parser.add_argument("--summary-md-out", type=Path, default=None, help="Output progression_summary.md.")
    parser.add_argument(
        "--observations-in",
        type=Path,
        default=None,
        help="Existing progression_results.jsonl with turn_observations; skips LLM and recomputes deterministic mapping.",
    )
    parser.add_argument("--brand", choices=("all", "foton", "unpk"), default="all")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--judge-mode", choices=("codex", "fake"), default="codex")
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--judge-reasoning", default="high")
    parser.add_argument("--timeout-sec", type=int, default=180)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--codex-isolated", dest="codex_isolated", action="store_true", default=True)
    parser.add_argument("--no-codex-isolated", dest="codex_isolated", action="store_false")
    args = parser.parse_args(argv)
    progression_llm_call_counter = JudgeCallCounter()
    setattr(args, "progression_llm_call_counter", progression_llm_call_counter)

    sim_input = sim.load_dynamic_sim_input(args.scenarios)
    persona_by_id = {str(item.get("dialog_id") or ""): item for item in sim_input.personas}
    dialogs = [
        dialog
        for dialog in sim.load_transcripts(args.transcripts)
        if args.brand == "all" or dialog.get("brand") == args.brand
    ]
    if args.limit > 0:
        dialogs = dialogs[: args.limit]

    observations_by_dialog = load_observations_by_dialog(args.observations_in)
    rows = assess_dialogs(
        args=args,
        dialogs=dialogs,
        persona_by_id=persona_by_id,
        observations_by_dialog=observations_by_dialog,
    )
    summary = summarize_results(rows, llm_calls=progression_llm_call_counter.snapshot())
    summary = {
        **summary,
        "transcripts": str(args.transcripts),
        "scenarios": str(args.scenarios),
        "judge_mode": "stored_observations" if observations_by_dialog else args.judge_mode,
        "model": "from_observations" if observations_by_dialog else args.model if args.judge_mode == "codex" else "fake",
        "observations_in": str(args.observations_in) if args.observations_in else "",
    }

    out = args.out or (args.transcripts.parent / "progression_results.jsonl")
    summary_out = args.summary_out or (out.parent / "progression_summary.json")
    summary_md_out = args.summary_md_out or (out.parent / "progression_summary.md")
    sim.write_jsonl(out, rows)
    summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    summary_md_out.write_text(render_summary_md(summary), encoding="utf-8")
    print(
        json.dumps(
            {
                "ok": True,
                "dialogs": len(rows),
                "turns": summary.get("turns"),
                "out": str(out),
                "summary_out": str(summary_out),
                "summary_md_out": str(summary_md_out),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
