"""Шаг [2]: детерминированный ретривал ЗНАЧЕНИЙ фактов по ключам из контракта.

Инвариант: значения берутся ТОЛЬКО из верифицированного бренд-скоупного склада confirmed_facts.
LLM (шаг 1) лишь НАЗВАЛ нужные ключи; цифры/даты/адреса подставляет этот слой. Так выдумка не может попасть на вход черновика.

Референс хранит склад как dict brand -> {fact_key: value}. Кодекс при переносе подключает реальный
client_safe_facts_<brand> (jsonl/snapshot) и достаёт значения по fact_key.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass
class RetrievalResult:
    facts: dict[str, str]        # key -> value (только найденное в складе активного бренда)
    missing: tuple[str, ...]     # ключи, которых в складе нет → честный узкий handoff по этой детали


def _norm_brand(b: str) -> str:
    b = (b or "").strip().lower()
    if b in {"foton", "фотон"}:
        return "foton"
    if b in {"unpk", "унпк", "унпк мфти"}:
        return "unpk"
    return b


def retrieve_facts(
    *, needed_fact_keys, active_brand: str, store: Mapping[str, Mapping[str, str]]
) -> RetrievalResult:
    """Достать значения нужных ключей ТОЛЬКО из склада активного бренда.

    store: {"foton": {key: value}, "unpk": {...}}. Кросс-бренд ключи недоступны (жёсткий бренд-скоуп).
    """
    brand = _norm_brand(active_brand)
    brand_store = store.get(brand) or {}
    facts: dict[str, str] = {}
    missing: list[str] = []
    for key in needed_fact_keys or ():
        key = str(key).strip()
        if not key:
            continue
        if key in brand_store and str(brand_store[key]).strip():
            facts[key] = str(brand_store[key])
        else:
            missing.append(key)
    return RetrievalResult(facts=facts, missing=tuple(dict.fromkeys(missing)))
