# Phase 12 · TZ-06 — миграция `matkap`-guarantee в v2

Снимок: HEAD `36e23cb8`. Приоритет диспетчера: 40. Зависит от: диспетчера.

## 1. Цель
По маткапиталу не обещать одобрение СФР и не принимать региональный маткапитал. Отвечать утверждёнными формулировками (рассмотрение — СФР; работаем с федеральным). Финансово-юр. обещание = риск.

## 2. Текущее состояние в legacy
- Селектор: `_matkap_safe_template(result, *, client_message, context)` — `subscription_llm.py:4446`. Применение: `:2370-2376` — flag `matkap_safe_template_applied`, checklist «Маткапитал: не обещать одобрение СФР…».
- Тексты: `MATKAP_REGIONAL_SAFE_TEXT:192`, `MATKAP_SFR_REVIEW_SAFE_TEXT:193`, `MATKAP_FEDERAL_TIMING_SAFE_TEXT:194`.
- **В v2 НЕ вызывается.**
- По CLAUDE.md: маткапитал можно объяснять справочно для обоих брендов (с active_brand). То есть P0-аспект — именно guarantee-блок (СФР/региональный), справочная часть — P1.

## 3. Точка вставки в v2
Диспетчер, priority 40 (финансовое обещание выше общих цен). Skip как legacy (`:2370`: не применять, если terminal). 

## 4. Зависимости (KB)
Тексты — константы; справочные matkap-факты (сроки СФР) уже в v6.3 (`matkap.timeline.*` — видел в транскриптах). `MATKAP_FEDERAL_TIMING_SAFE_TEXT` содержит сроки — проверить, что числа в нём в whitelist `_is_verified_safe_numeric_template:1631`. **Новых фактов не нужно.** Блок Г: OK.

## 5. Точная правка
```python
_MATKAP_SPEC = TemplateSpec(
    name="matkap", priority=40,
    produce=lambda r, cm, ctx: _matkap_safe_template(r, client_message=cm, context=ctx),
    route_on_apply="keep_or_draft",  # manager_only→manager_only иначе draft_for_manager (как :2372)
    flag="matkap_safe_template_applied",
    checklist="Маткапитал: не обещать одобрение СФР, не принимать региональный.",
)
```

## 6. Регрессы
Positive:
1. `«одобрят маткапитал?»` → MATKAP_SFR_REVIEW_SAFE_TEXT («рассмотрение проводит СФР, не обещаем»).
2. `«примете региональный маткапитал?»` → MATKAP_REGIONAL_SAFE_TEXT («только федеральный»).
3. `«за сколько проходит маткапитал?»` → MATKAP_FEDERAL_TIMING (сроки из факта).
Контрольные negative:
4. `«можно оплатить маткапиталом?»` (справочно, без обещания одобрения) → справочный ответ (CLAUDE.md разрешает), не жёсткий отказ.
5. Обычная оплата (не маткап) → matkap НЕ срабатывает.
6. Числа сроков в MATKAP_FEDERAL_TIMING → не ловятся unsupported_promise (whitelist).
7. active_brand учтён (маткап для обоих брендов, но в рамках текущего).

## 7. Backward compatibility
- Legacy matkap тесты зелёные.
- Справочная часть маткапитала (разрешена CLAUDE.md) не зарезается.

## 8. Открытые вопросы Кодексу
1. Разделить P0 (guarantee-блок СФР/региональный) и P1 (справочные сроки) — мигрировать сейчас весь `_matkap_safe_template` или только guarantee-ветки? Рекомендую целиком (один селектор), guarantee — критичная часть.
2. `MATKAP_FEDERAL_TIMING_SAFE_TEXT` числа → whitelist в `_is_verified_safe_numeric_template`.
