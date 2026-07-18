from __future__ import annotations

"""Слой «человечности» — детерминированные гварды поверх генерации (референс).

Закрывает остаток round-5 (PASS_WITH_NOTES): повторы, игнор нового вопроса,
лишний хендофф при наличии факта, мета-реплики клиенту. Это НЕ генерация —
это проверки на выходе/маршруте, которые ВЫЧИТАЮТ (блок/пометка регенерации),
а не переписывают (в духе всего проекта).

ГРАНИЦЫ ПРИМЕНЕНИЯ (поправки Кодекса 2026-05-25, подтверждаю):
- Это ФИНАЛЬНАЯ страховка. Гварды НЕ должны перехватывать ответ ДО специализированных
  проверенных шаблонов (цена/пробное/лагерь) — иначе нарушается класс «более точный ответ
  нельзя перезаписывать» (см. held/decision_policy). Применять последним слоем.
- НЕ применять к сухим P0-хендоффам и к non_question/context_update/wait_for_more
  (благодарность/завершение): там повтор и «сухой» текст ДОПУСТИМЫ. is_near_repeat и
  has_meta_leak вызывать только на обычных содержательных клиентских ответах.

Зависит только от stdlib. Принимает простые типы (строки/множества), чтобы
Кодекс легко вшил на стадии пост-обработки черновика.
"""

import difflib
import re
from typing import Sequence

from mango_mvp.channels.output_verification_floor import (
    _META_CLIENT_MARKERS,
    _META_FACT_PHRASE_RE,
    _WORD_CHARS,
    _has_meta_fact_phrase,
    _norm,
    _tokens,
    has_meta_leak,
    is_near_repeat,
    repeat_ratio,
)


HUMANITY_GUARDS_SCHEMA_VERSION = "humanity_guards_ref_v1_2026_05_25"


# Мета/служебные фразы, которые НЕ должны попадать клиенту (manager-facing/внутреннее).
# ВАЖНО (поправка Кодекса 2026-05-25): сюда входит ТОЛЬКО действительно внутренний/служебный
# текст. НЕ включать «приняли обращение» и «по проверенным данным» — это шаблонные зачины/тон
# (их ловит is_near_repeat/templated-слой), иначе has_meta_leak ложно сработает на ЗАКОННОМ
# сухом P0-хендоффе и сломает безопасный P0-ответ.


def meta_markers_present(text: object) -> list[str]:
    low = _norm(text)
    low_raw = str(text or "").casefold()
    markers = [m for m in _META_CLIENT_MARKERS if (m in low) or (m in low_raw)]
    if _has_meta_fact_phrase(text):
        markers.append("fact_phrase_leak")
    return markers


def should_answer_not_handoff(
    *,
    p0_required: bool,
    has_retrieved_answer_fact: bool,
    route: str,
) -> bool:
    """True → текущий маршрут в менеджера НЕОБОСНОВАН (есть факт, нет P0) → надо отвечать.
    Не ослабляет P0: при p0_required всегда False."""
    if p0_required:
        return False
    routed_to_manager = str(route or "").strip() in {"manager_only", "draft_for_manager"}
    return routed_to_manager and has_retrieved_answer_fact


def humanity_route_action(
    *,
    p0_required: bool,
    has_retrieved_answer_fact: bool,
    route: str,
    message_type: str = "question",
    direct_question_answered: bool = True,
) -> dict:
    """ДЕЙСТВЕННОЕ решение слоя человечности (а НЕ сигнал-no-op).

    Возвращает {route, regenerate, reason}. Кодекс применяет: если regenerate=True —
    перегенерировать ответ из факта без хеджа; route — итоговый маршрут.
    Правила:
    - P0: не трогаем (безопасность важнее) → как есть, regenerate=False.
    - non_question/context_update/wait_for_more (благодарность/завершение): не вмешиваемся.
    - иначе если факт ИЗВЛЕЧЁН и маршрут в менеджера без P0 → route='bot_answer_self',
      regenerate=True ('ответь из факта, не хеджи');
    - иначе если прямой вопрос НЕ закрыт → regenerate=True (ответить на дельту), route как есть.
    """
    if p0_required:
        return {"route": route, "regenerate": False, "reason": "p0_keep"}
    if str(message_type) in {"non_question", "context_update", "wait_for_more", "manager_only"}:
        return {"route": route, "regenerate": False, "reason": "non_question_keep"}
    routed_to_manager = str(route or "").strip() in {"manager_only", "draft_for_manager"}
    if has_retrieved_answer_fact and routed_to_manager:
        return {"route": "bot_answer_self", "regenerate": True, "reason": "answer_from_fact_not_handoff"}
    if not direct_question_answered:
        return {"route": route, "regenerate": True, "reason": "answer_the_delta"}
    return {"route": route, "regenerate": False, "reason": "ok"}


def unanswered_direct_question(
    client_message: object,
    draft_text: object,
    *,
    client_topics: Sequence[str] = (),
    draft_topics: Sequence[str] = (),
) -> bool:
    """Клиент задал прямой вопрос (с темой), а черновик не покрывает НИ одну из тем
    вопроса → вопрос проигнорирован. Эвристика-сигнал для регенерации, не приговор."""
    if "?" not in str(client_message or ""):
        return False
    ct = {str(t) for t in client_topics if str(t or "").strip()}
    if not ct:
        return False
    dt = {str(t) for t in draft_topics if str(t or "").strip()}
    return not (ct & dt)
