"""Оркестратор v2 (полный поток, цельная архитектура, под-тумблеры для изоляции при отладке).

Поток на ход:
  [0] P0 пре-гейт + [1] понять → контракт-план  → manager_only/P0: сухой P0 / безопасный fallback, STOP
  [2] ретривал значений по ключам (из контракта)
  [3] черновик: контракт + client_state + СТИЛЕВЫЕ примеры + ТОЛЬКО sourced-слоты + facts[]  (свободно, живо)
  [4] ЖЁСТКИЙ verifier (бренд/числа-заземление/forbidden/мета/P0) — output_verifier
  [5] СЕМАНТИЧЕСКАЯ верность (не-числовые утверждения) — quality_layer.check_claim_faithfulness
  [4-5 fail] → ремонт по конкретным замечаниям (repair_fn), ≤2, ре-проверка; не вышло → безопасный fallback
  [6] МЯГКАЯ проверка формы (штамп/повтор/нет шага/канцелярит) — quality_layer.form_check
  [6 issues] → X2-ТЕПЛО (warmth_fn) → РЕ-проверка [4]+[5]; прошло → берём тёплый, иначе оставляем верифицированный черновик
  → финал

Под-тумблеры (Toggles): enforce_slot_evidence, semantic_faithfulness, form_warmth. Любой LLM None = соответствующий шаг выключен.
Безопасность держится на: значения только из склада + жёсткий verifier + семантическая верность + ре-проверка тепла + fallback.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

from dialogue_understanding import understand, AnswerContract
from fact_provider import retrieve_facts
from output_verifier import verify, passed
from quality_layer import check_claim_faithfulness, form_check, warmth_rewrite

MAX_REPAIR_ATTEMPTS = 2


@dataclass
class Toggles:
    enforce_slot_evidence: bool = True     # в черновик только sourced-слоты
    semantic_faithfulness: bool = True     # §11.3 LLM-проверка смысловых утверждений
    form_warmth: bool = True               # §12 form-check + X2-тепло


@dataclass
class V2Result:
    draft_text: str
    route: str
    manager_only: bool
    contract: AnswerContract
    facts: dict = field(default_factory=dict)
    missing: tuple = ()
    hard_findings: list = field(default_factory=list)
    unsupported_claims: list = field(default_factory=list)
    form_findings: list = field(default_factory=list)
    warmed: bool = False
    repaired: bool = False
    fallback_reason: str = ""


def _dry_p0() -> str:
    return "Приняли обращение. Передам его ответственному сотруднику, он вернётся с ответом."


def _fallback(contract: AnswerContract) -> str:
    return f"Передам менеджеру уточнить именно это: {contract.current_question or 'ваш вопрос'}. Он подтвердит точную информацию."


def build_draft_prompt(*, conversation, contract: AnswerContract, facts: Mapping[str, str],
                       missing: Sequence[str], style_examples: Sequence[str] = ()) -> str:
    hist = "\n".join(f"{m.get('role','?')}: {m.get('text','')}" for m in conversation)
    facts_block = "\n".join(f"- {k}: {v}" for k, v in facts.items()) or "(нет подтверждённых фактов)"
    subq = "\n".join(f"- {s.text} [{s.answerable}]" for s in contract.subquestions) or f"- {contract.current_question}"
    slots = contract.assertable_slots() if True else {}
    return (
        f"Бренд: {contract.active_brand} (другой бренд не упоминать). Ситуация клиента: {contract.client_state or 'обычная'} "
        "(подстрой тон, НЕ называй эмоцию вслух).\n"
        f"Под-вопросы (ответь на каждый по сути):\n{subq}\n"
        + (f"Клиент ОТРИЦАЕТ (не отвечать про это): {', '.join(contract.denied_topics)}\n" if contract.denied_topics else "")
        + (f"Известно из диалога (можно использовать): {slots}\n" if slots else "")
        + (f"НЕ утверждай неназванное: используй ТОЛЬКО известное выше и факты ниже.\n")
        + f"ПОДТВЕРЖДЁННЫЕ ФАКТЫ (единственный источник конкретики):\n{facts_block}\n"
        + (f"НЕТ факта по: {', '.join(missing)} → честно скажи, что уточнит менеджер; НЕ подставляй соседний факт.\n" if missing else "")
        + (f"Нельзя подставлять: {', '.join(contract.forbidden_substitutions)}\n" if contract.forbidden_substitutions else "")
        + (f"СТИЛЬ (живые примеры по теме — манера, НЕ источник фактов):\n" + "\n".join(f"  • {e}" for e in style_examples) + "\n" if style_examples else "")
        + "Правила: живо и по-человечески; только факты выше; не раскрывай ИИ; не обещай возврат/результат; один мягкий следующий шаг.\n"
        + f"ДИАЛОГ:\n{hist}\nНапиши черновик ответа клиенту."
    )


def _hard_check(draft, *, facts, contract, client_words, faithfulness_fn, toggles):
    """Вернуть (findings, unsupported, semantic_available). semantic_available=False → проверка не отработала → fail-closed."""
    findings = verify(draft, facts=facts, active_brand=contract.active_brand,
                      denied_topics=contract.denied_topics, forbidden_substitutions=contract.forbidden_substitutions)
    unsupported: list = []
    semantic_available = True
    if toggles.semantic_faithfulness:
        fr = check_claim_faithfulness(draft, facts=facts, client_words=client_words, faithfulness_fn=faithfulness_fn)
        unsupported = list(fr.unsupported)
        semantic_available = fr.available
    return findings, unsupported, semantic_available


def run_pipeline(
    *,
    conversation: Sequence[Mapping[str, str]],
    active_brand: str,
    fact_store: Mapping[str, Mapping[str, str]],
    catalog: Sequence[str],
    understand_fn: Callable[[str], object] | None,
    draft_fn: Callable[[str], str] | None,
    repair_fn: Callable[[str], str] | None = None,
    faithfulness_fn: Callable[[str], object] | None = None,
    warmth_fn: Callable[[str], str] | None = None,
    style_examples: Sequence[str] = (),
    toggles: Toggles | None = None,
) -> V2Result:
    toggles = toggles or Toggles()
    client_words = str(conversation[-1].get("text") or "") if conversation else ""

    # [0]+[1]
    contract = understand(conversation=conversation, active_brand=active_brand,
                          fact_key_catalog=catalog, understand_fn=understand_fn)
    if contract.manager_only():
        return V2Result(draft_text=_dry_p0() if contract.is_p0 else _fallback(contract),
                        route="manager_only", manager_only=True, contract=contract,
                        fallback_reason="p0" if contract.is_p0 else "contract_manager_only")

    # [2] ретривал
    rr = retrieve_facts(needed_fact_keys=contract.all_needed_fact_keys(), active_brand=active_brand, store=fact_store)

    # [3] черновик
    if draft_fn is None:
        return V2Result(draft_text=_fallback(contract), route="draft_for_manager", manager_only=False,
                        contract=contract, facts=rr.facts, missing=rr.missing, fallback_reason="no_draft_fn")
    prompt = build_draft_prompt(conversation=conversation, contract=contract, facts=rr.facts,
                                missing=rr.missing, style_examples=style_examples)
    try:
        draft = str(draft_fn(prompt) or "").strip()
    except Exception:
        draft = ""
    if not draft:
        return V2Result(draft_text=_fallback(contract), route="draft_for_manager", manager_only=False,
                        contract=contract, facts=rr.facts, missing=rr.missing, fallback_reason="draft_error")

    # [4]+[5] жёсткая + семантическая проверка, [4-5 fail] ремонт
    repaired = False
    findings, unsupported, sem_ok = _hard_check(draft, facts=rr.facts, contract=contract, client_words=client_words,
                                                faithfulness_fn=faithfulness_fn, toggles=toggles)
    if not sem_ok:   # FAIL-CLOSED: проверку не удалось выполнить → не отдаём автономно
        return V2Result(draft_text=_fallback(contract), route="draft_for_manager", manager_only=False,
                        contract=contract, facts=rr.facts, missing=rr.missing, fallback_reason="semantic_check_unavailable")
    attempts = 0
    while (findings or unsupported) and repair_fn is not None and attempts < MAX_REPAIR_ATTEMPTS:
        attempts += 1
        instr = "; ".join([f.detail for f in findings] + [f"неподтверждённое утверждение: {u}" for u in unsupported])
        try:
            candidate = str(repair_fn(_repair_prompt(draft, instr, rr.facts)) or "").strip()
        except Exception:
            break
        if not candidate:
            break
        draft, repaired = candidate, True
        findings, unsupported, sem_ok = _hard_check(draft, facts=rr.facts, contract=contract, client_words=client_words,
                                                    faithfulness_fn=faithfulness_fn, toggles=toggles)
        if not sem_ok:
            return V2Result(draft_text=_fallback(contract), route="draft_for_manager", manager_only=False,
                            contract=contract, facts=rr.facts, missing=rr.missing, repaired=repaired,
                            fallback_reason="semantic_check_unavailable")
    if findings or unsupported:
        return V2Result(draft_text=_fallback(contract), route="draft_for_manager", manager_only=False,
                        contract=contract, facts=rr.facts, missing=rr.missing, hard_findings=list(findings),
                        unsupported_claims=list(unsupported), repaired=repaired, fallback_reason="hard_verification_failed")

    # [6] мягкая форма → X2-тепло (с РЕ-проверкой жёстких)
    form_findings = []
    warmed = False
    if toggles.form_warmth:
        prev = [m.get("text", "") for m in conversation if m.get("role") == "bot"]
        form_findings = form_check(draft, previous_bot_texts=prev)
        if form_findings and warmth_fn is not None:
            warm = warmth_rewrite(draft, client_state=contract.client_state,
                                  form_issues=[f.code for f in form_findings], facts=rr.facts, warmth_fn=warmth_fn)
            if warm:
                wf, wu, wsem = _hard_check(warm, facts=rr.facts, contract=contract, client_words=client_words,
                                           faithfulness_fn=faithfulness_fn, toggles=toggles)
                if wsem and not wf and not wu:  # тёплый прошёл ВСЕ проверки → берём; иначе оставляем верифицированный исходный
                    draft, warmed = warm, True

    return V2Result(draft_text=draft, route="bot_answer_self", manager_only=False, contract=contract,
                    facts=rr.facts, missing=rr.missing, form_findings=[f.code for f in form_findings],
                    warmed=warmed, repaired=repaired)


def _repair_prompt(draft, instr, facts):
    facts_block = "\n".join(f"- {k}: {v}" for k, v in facts.items()) or "(нет фактов)"
    return (
        "Исправь РОВНО это (содержание/факты), смысл и маршрут не меняй, новых фактов вне списка не вводи:\n"
        f"ЗАМЕЧАНИЯ: {instr}\nФАКТЫ:\n{facts_block}\nЧЕРНОВИК:\n{draft}\nВерни только исправленный текст."
    )
