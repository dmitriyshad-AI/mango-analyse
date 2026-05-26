"""Слой качества v2 (ТЗ §11.3 + §12): семантическая верность фактам, проверка формы, X2-тепло.

ДВА разных назначения, не путать:
- check_claim_faithfulness — ЖЁСТКАЯ проверка СМЫСЛОВЫХ утверждений (не только чисел): «по будням», «есть пробное»,
  «запись сохраняется» — каждое конкретное утверждение должно быть в facts[] ИЛИ в словах клиента. Нарушение = выдумка → ремонт/фоллбэк.
  Открыто детерминированно не решается → отдельный фокусный LLM-проход (faithfulness_fn). Это второй вызов на ПРОВЕРКУ, не на генерацию.
- form_check — МЯГКАЯ проверка формы (штамп/повтор/нет шага/канцелярит) → НЕ блок, а ТРИГГЕР X2-тепла.
- warmth_rewrite — X2 «сделай теплее»: меняет ФОРМУ, не содержание; выход обязан снова пройти жёсткие проверки (в pipeline).
"""

from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence


# ---------- §11.3 семантическая верность (ЖЁСТКО, LLM-инъекция) ----------
def build_faithfulness_prompt(draft: str, *, facts: Mapping[str, str], client_words: str) -> str:
    facts_block = "\n".join(f"- {k}: {v}" for k, v in facts.items()) or "(фактов нет)"
    return (
        "Проверь черновик ответа на ВЕРНОСТЬ. Верни СТРОГО JSON: {\"unsupported\": [<конкретные утверждения, которых НЕТ "
        "ни в ФАКТАХ, ни в словах клиента>]}.\n"
        "Конкретное утверждение = расписание/дни, формат, наличие (пробное/места/запись), сроки, условия, цены, действия.\n"
        "НЕ считай нарушением общую вежливость и предложение помочь.\n"
        f"ФАКТЫ:\n{facts_block}\n"
        f"СЛОВА КЛИЕНТА:\n{client_words}\n"
        f"ЧЕРНОВИК:\n{draft}\n"
        "Только JSON."
    )


@dataclass
class FaithfulnessResult:
    unsupported: list[str]
    available: bool          # False = проверка НЕ отработала (сбой/мусор) → fail-CLOSED в pipeline


def check_claim_faithfulness(
    draft: str, *, facts: Mapping[str, str], client_words: str,
    faithfulness_fn: Callable[[str], object] | None,
) -> FaithfulnessResult:
    """Проверка смысловых утверждений. FAIL-CLOSED: если проверка не отработала (сбой/мусор), available=False —
    pipeline должен трактовать это как «не верифицировано» → НЕ отдавать автономно (узкий хендофф/менеджер), а НЕ PASS.
    faithfulness_fn=None означает «семантика выключена тумблером» (available=True, на неё не опираемся)."""
    if faithfulness_fn is None:
        return FaithfulnessResult(unsupported=[], available=True)
    prompt = build_faithfulness_prompt(draft, facts=facts, client_words=client_words)
    try:
        raw = faithfulness_fn(prompt)
    except Exception:
        return FaithfulnessResult(unsupported=[], available=False)  # сбой → fail-closed
    data: object = raw
    if isinstance(raw, str):
        try:
            data = json.loads(raw)
        except Exception:
            return FaithfulnessResult(unsupported=[], available=False)  # кривой JSON → fail-closed
    if isinstance(data, Mapping) and "unsupported" in data:
        items = data.get("unsupported") or []
    elif isinstance(data, (list, tuple)):
        items = data
    else:
        return FaithfulnessResult(unsupported=[], available=False)  # неожиданная форма → fail-closed
    return FaithfulnessResult(unsupported=[str(x).strip() for x in items if str(x).strip()], available=True)


# ---------- §12.4 проверка ФОРМЫ (МЯГКО → триггер тепла) ----------
_STOCK_OPENERS = ("сориентирую по проверенным данным", "по проверенным данным")
_CLERICAL = ("осуществляется", "в рамках", "по вопросу о", "данный", "необходимо отметить", "вышеуказанн")


def _norm(t: str) -> str:
    return re.sub(r"\s+", " ", str(t or "").strip().lower())


@dataclass
class FormFinding:
    code: str
    detail: str


def form_check(draft: str, *, previous_bot_texts: Sequence[str] = ()) -> list[FormFinding]:
    out: list[FormFinding] = []
    low = _norm(draft)
    if any(low.startswith(c) or c in low[:40] for c in _STOCK_OPENERS):
        out.append(FormFinding("stock_opener", "канцелярский штамп-зачин"))
    for prev in previous_bot_texts:
        p = _norm(prev)
        if len(p) > 25 and len(low) > 25 and difflib.SequenceMatcher(None, p, low).ratio() > 0.85:
            out.append(FormFinding("near_repeat", "почти дословный повтор предыдущего ответа"))
            break
    if not re.search(r"[?]|подобрать|подскаж|помоч|следующий шаг|записать|уточн", low):
        out.append(FormFinding("no_next_step", "нет мягкого следующего шага"))
    if any(c in low for c in _CLERICAL):
        out.append(FormFinding("clerical", "канцелярит"))
    return out


# ---------- §12.5 X2-тепло (меняет форму, не содержание) ----------
def build_warmth_prompt(draft: str, *, client_state: str, form_issues: Sequence[str], facts: Mapping[str, str]) -> str:
    facts_block = "\n".join(f"- {k}: {v}" for k, v in facts.items()) or "(нет фактов)"
    return (
        "Перепиши ответ ЖИВЕЕ и теплее, СОХРАНив весь смысл и факты. Меняй только ФОРМУ.\n"
        f"Ситуация клиента: {client_state or 'обычная'} (подстрой регистр; НЕ называй эмоцию вслух).\n"
        f"Что поправить по форме: {', '.join(form_issues) or 'тон/прямота'}.\n"
        "ЖЁСТКО: не вводи новых чисел/дат/имён/условий вне фактов; не упоминай другой бренд; не раскрывай ИИ; не обещай возврат/результат.\n"
        "Сначала прямой ответ, потом 1-2 пояснения, один мягкий следующий шаг. Без штампов и канцелярита.\n"
        f"ФАКТЫ (источник конкретики):\n{facts_block}\n"
        f"ОТВЕТ:\n{draft}\n"
        "Верни только переписанный текст."
    )


def warmth_rewrite(
    draft: str, *, client_state: str, form_issues: Sequence[str], facts: Mapping[str, str],
    warmth_fn: Callable[[str], str] | None,
) -> str | None:
    """Вернуть тёплый кандидат или None (выключено/сбой). Ре-верификацию делает pipeline."""
    if warmth_fn is None:
        return None
    prompt = build_warmth_prompt(draft, client_state=client_state, form_issues=form_issues, facts=facts)
    try:
        candidate = str(warmth_fn(prompt) or "").strip()
    except Exception:
        return None
    return candidate or None
