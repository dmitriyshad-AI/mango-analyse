from __future__ import annotations

"""Отбор фактов в confirmed_facts — референс слоя ИЗВЛЕЧЕНИЯ (recall-first).

Зачем (корень потолка качества, см. D1_fact_retrieval_findings/CEILING_FINDINGS):
    Сейчас факты подбираются по маркерам ТЕКУЩЕЙ реплики + scope, с капом 10, и
    нужный факт-ОТВЕТ часто не попадает в confirmed_facts, хотя ЕСТЬ в базе:
      - факт без scope-маркера отбрасывается (fact_scope_spec:138);
      - факт-ответ под «соседним» ключом не матчится (за год 14% лежит под
        payment_options..., а запрос — discounts.current);
      - на follow-up маркеров нет → confirmed_facts схлопывается в 0;
      - факт-ответ может оказаться ниже соседних в топ-10 и срезаться капом.
    Промпт при этом ЗАПРЕЩАЕТ называть числа вне confirmed_facts — поэтому промах
    извлечения ВЫНУЖДАЕТ бота увиливать/уходить в менеджера.

Этот модуль чинит ИМЕННО recall, не ослабляя точность:
    1. факт-ОТВЕТ (совпал с required_fact_keys по смысловому ключу) ГАРАНТИРОВАННО
       включается, даже если кап 10 его срезал бы;
    2. сопоставление ключей — по смысловым алиасам, а не точному префиксу;
    3. факт без scope НЕ отбрасывается только из-за отсутствия scope;
    4. факт ЧУЖОГО продукта/формата (в blocked_scopes) по-прежнему исключается
       (точность сохранена: город↔выезд, олимп↔обычный онлайн не путаем);
    5. тема извлечения берётся из ролей ИЛИ из held (follow-up не обнуляет тему).

Чистый stdlib. Кандидат-факт — это dict с полями fact_key, brand, text, scopes(set).
Кодекс подаёт сюда реальные client-safe факты бренда.
"""

from typing import Mapping, Sequence


FACT_RETRIEVAL_SCHEMA_VERSION = "fact_retrieval_ref_v1_2026_05_25"

# Смысловые алиасы: голова required_fact_key -> подстроки, которые считаются совпадением
# в fact_key. Чинит таксономию (за год 14% под payment_options, адрес под locations_*.address).
_KEY_ALIASES: dict[str, tuple[str, ...]] = {
    "prices": ("price", "prices", "tuition", "year", "semester", "cost", "стоим"),
    "discounts": ("discount", "discounts"),
    "installment_terms": ("installment", "dolyami", "payment_options", "rassroch"),
    "trial_class": ("trial", "fragment", "фрагмент", "probn"),
    "programs": ("program", "camp", "ls_city", "lvsh", "senior_school", "direction", "subject", "olympiad"),
    "availability": ("availability", "seats", "places", "mest"),
    "schedule": ("schedule", "weekly_lessons", "available_schedules", "days", "lesson"),
    "formats": ("format", "online", "offline", "ochno"),
    "locations": ("location", "address", "addresses"),
    "documents": ("document", "spravka", "certificate", "cert"),
    "matkap_documents": ("matkap", "sfr", "materin"),
    "tax_deduction_procedure": ("tax", "deduction", "vychet", "ndfl", "fns"),
}


def _key_head(required_key: str) -> str:
    return str(required_key or "").split(".")[0].strip().casefold()


def key_matches(required_key: str, fact_key: str) -> bool:
    head = _key_head(required_key)
    if not head:
        return False
    fk = str(fact_key or "").casefold()
    subs = _KEY_ALIASES.get(head, (head,))
    return any(s in fk for s in subs)


def select_confirmed_facts(
    candidates: Sequence[Mapping[str, object]],
    *,
    active_brand: str = "",
    required_fact_keys: Sequence[str] = (),
    active_topics: Sequence[str] = (),
    blocked_scopes: Sequence[str] = (),
    k: int = 10,
) -> list[Mapping[str, object]]:
    """Вернуть confirmed_facts: гарантированно с фактом-ответом, без чужой области.

    required_fact_keys/active_topics — текущие ИЛИ удержанные (held.retrieval_context()).
    """
    req = [str(x) for x in required_fact_keys if str(x or "").strip()]
    topics = {str(x) for x in active_topics if str(x or "").strip()}
    blocked = {str(x) for x in blocked_scopes if str(x or "").strip()}
    brand = str(active_brand or "").strip()

    answer: list[Mapping[str, object]] = []
    rest: list[tuple[int, Mapping[str, object]]] = []

    for f in candidates:
        fb = str(f.get("brand") or "").strip()
        if brand and fb and fb != brand:
            continue  # чужой бренд — никогда
        fscopes = {str(s) for s in (f.get("scopes") or [])}
        if fscopes & blocked:
            continue  # чужой продукт/формат — исключаем (точность сохранена)
        fk = str(f.get("fact_key") or "")
        is_answer = any(key_matches(rk, fk) for rk in req)
        if is_answer:
            answer.append(f)
            continue
        score = 0
        if fscopes & topics:
            score += 10
        # факт без scope НЕ штрафуем (recall): он может быть релевантным ответом
        rest.append((score, f))

    rest.sort(key=lambda x: -x[0])
    rest_facts = [f for _, f in rest]
    # факт-ОТВЕТ всегда внутри; добиваем релевантными до k
    result = answer + rest_facts
    limit = max(k, len(answer))
    return result[:limit]
