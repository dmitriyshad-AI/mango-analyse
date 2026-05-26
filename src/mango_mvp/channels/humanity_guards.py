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


HUMANITY_GUARDS_SCHEMA_VERSION = "humanity_guards_ref_v1_2026_05_25"

_WORD_CHARS = "0-9a-zа-я"


def _norm(text: object) -> str:
    s = str(text or "").casefold().replace("ё", "е")
    s = re.sub(rf"[^{_WORD_CHARS} ]", " ", s)
    return " ".join(s.split())


def _tokens(text: object) -> set[str]:
    return set(_norm(text).split())


def repeat_ratio(a: object, b: object) -> float:
    """Сходство двух реплик (0..1). 1.0 = практически одно и то же."""
    ta, tb = _tokens(a), _tokens(b)
    jaccard = len(ta & tb) / len(ta | tb) if ta and tb else 0.0
    a_norm, b_norm = _norm(a), _norm(b)
    sequence = 0.0
    if len(a_norm) >= 25 and len(b_norm) >= 25:
        sequence = difflib.SequenceMatcher(None, a_norm, b_norm).ratio()
    return max(jaccard, sequence)


def is_near_repeat(draft: object, prior_bot_texts: Sequence[object], *, threshold: float = 0.8) -> bool:
    """Черновик почти дублирует один из ПРЕДЫДУЩИХ ответов бота → повтор.
    Применять на ВСЕХ ветках (включая P0/handoff), кроме чисто служебного сухого
    P0-хендоффа на ПОВТОРНОЙ P0-реплике."""
    d = _norm(draft)
    if len(d.split()) < 4:
        return False
    return any(repeat_ratio(draft, p) >= threshold for p in prior_bot_texts)


# Мета/служебные фразы, которые НЕ должны попадать клиенту (manager-facing/внутреннее).
# ВАЖНО (поправка Кодекса 2026-05-25): сюда входит ТОЛЬКО действительно внутренний/служебный
# текст. НЕ включать «приняли обращение» и «по проверенным данным» — это шаблонные зачины/тон
# (их ловит is_near_repeat/templated-слой), иначе has_meta_leak ложно сработает на ЗАКОННОМ
# сухом P0-хендоффе и сломает безопасный P0-ответ.
_META_CLIENT_MARKERS: tuple[str, ...] = (
    "автономный ответ",
    "автономный ответ не требуется",
    "если менеджер решит",
    "безопасный вариант",
    "без служебных пометок",
    "не оформляю как жалобу",
    "не оформляю как заявление",
    "не буду оформлять это как",
    "передам ему контекст диалога",
    "не требует автономного",
    "manager_only",
    "client_safe",
    "fact:",
    "fact_id",
    "trace_id",
    "source_id",
)


def has_meta_leak(text: object) -> bool:
    """Клиенту виден внутренний/служебный текст."""
    low = _norm(text)
    low_raw = str(text or "").casefold()
    return any((m in low) or (m in low_raw) for m in _META_CLIENT_MARKERS)


def meta_markers_present(text: object) -> list[str]:
    low = _norm(text)
    low_raw = str(text or "").casefold()
    return [m for m in _META_CLIENT_MARKERS if (m in low) or (m in low_raw)]


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
