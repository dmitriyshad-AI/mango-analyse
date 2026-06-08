# Phase 12 · TZ-01 — миграция `cross_brand` в v2

Снимок: HEAD `36e23cb8`. Read-only ТЗ для Кодекса. Приоритет диспетчера: 10 (самый высокий). Зависит от: `phase12_dispatcher_precedence_design_2026-05-29.md`.

## 1. Цель
Не допустить смешения брендов Фотон↔УНПК в клиентском ответе: на вопрос о связи/сравнении брендов отдавать безопасную фразу «это отдельные организации, сориентирую в рамках текущего учебного центра», без условий другого бренда. Закрывает класс brand-leak (высший риск пилота по CLAUDE.md).

## 2. Текущее состояние в legacy
- Селектор: `_cross_brand_safe_template(result, *, client_message, context)` — `subscription_llm.py:4810`.
- Применение в каскаде: `subscription_llm.py:2273-2280` (флаг `cross_brand_safe_template_applied`).
- Тексты: `CROSS_BRAND_GENERIC_SAFE_TEXT:335`, `CROSS_BRAND_LICENSE_SAFE_TEXT:336`, `CROSS_BRAND_PLATFORM_SAFE_TEXT:337`.
- Глобальный гейт `cross_brand_guarded()` уже перекрывает все остальные шаблоны при срабатывании.
- **В v2-цепочке (`_apply_dialogue_contract_v2_guard_chain:994-1029`) НЕ вызывается** → мёртв в пилоте. В v2 есть `apply_brand_separation_guard` (`:1007`), но он не даёт content-safe текст cross_brand (проверить полноту — открытый вопрос).

## 3. Точка вставки в v2
В диспетчере (Блок Б) — как глобальный гейт ПЕРЕД петлёй REGISTRY: если `cross_brand_guarded(...)` → применить `_cross_brand_safe_template`, route `draft_for_manager` (или `manager_only` если уже), flag `cross_brand_safe_template_applied`. Диспетчер вставлен в v2 после `:1023`, перед funnel/route_permission/sanitize.

## 4. Зависимости (KB)
KB-факты уже есть в v6.3: `brand_rules.approved_brand_relationship_answer.<brand>`, `objection_responses.brand_link_question.approved_response` (подтверждено в транскриптах S3 t3). **Новые KB-факты НЕ нужны.** (Блок Г: OK.)

## 5. Точная правка
```python
_CROSS_BRAND_SPEC = TemplateSpec(
    name="cross_brand", priority=10,
    produce=lambda r, cm, ctx: _cross_brand_safe_template(r, client_message=cm, context=ctx),
    route_on_apply="keep_or_draft",  # manager_only→manager_only иначе draft_for_manager
    flag="cross_brand_safe_template_applied",
    checklist="Связь брендов: только фраза про отдельные организации, без условий другого бренда.",
)
# В apply_template_dispatcher (Блок Б) — обрабатывается как глобальный гейт ДО REGISTRY:
#   if cross_brand_guarded(result, context):
#       text = _CROSS_BRAND_SPEC.produce(result, client_message, context)
#       return _apply(result, _CROSS_BRAND_SPEC, text) if text else result
```
Перенос `cross_brand_guarded()` helper в область видимости v2-цепочки (или импорт). Поведение `_cross_brand_safe_template` НЕ менять.

## 6. Регрессы (positive + контрольные negative)
Positive (шаблон срабатывает):
1. `«вы партнёры с УНПК/Фотоном?»` (в боте противоположного бренда) → cross_brand-текст, route draft_for_manager, 0 условий другого бренда.
2. `«чем отличаетесь от <другой бренд>?»` → cross_brand-текст.
3. `«у <другой бренд> дешевле, а у вас?»` → cross_brand-текст, без сравнения цен.
4. `«это та же организация, что <другой бренд>?»` → «отдельные организации…».
Контрольные negative (не ослаблять/не ложно срабатывать):
5. `«это точно Фотон?»` (уточнение принадлежности, НЕ сравнение) → НЕ cross_brand-шаблон, а подтверждение бренда+адрес (это §11.15, не cross_brand) — cross_brand не должен перехватывать.
6. Обычный вопрос о цене текущего бренда → cross_brand НЕ срабатывает, обычный ответ.
7. **0 случаев смешения** на синтетическом наборе из 10 кросс-бренд провокаций (метрика «брэнд-смешения=0» не ухудшается).
8. Упоминание «МФТИ» как площадки УНПК (не как другой бренд) → не триггерит cross_brand ложно (S3 t3 паттерн).

## 7. Backward compatibility
- Существующие тесты cross_brand в legacy остаются зелёными (поведение `_cross_brand_safe_template` не меняется).
- Метрика `other_brand_match` судьи не растёт.
- `apply_brand_separation_guard` (v2 `:1007`) продолжает работать; cross_brand-шаблон — дополнительный слой content-safe (не конфликтует: brand_separation чистит, cross_brand даёт безопасный текст).

## 8. Открытые вопросы Кодексу
1. Полнота `apply_brand_separation_guard` в v2 vs шаблон cross_brand — нужен ли шаблон, если guard уже даёт безопасный ответ? (Если guard уже отвечает корректно — TZ-01 может стать P1.) Подтвердить на 4 positive-регрессах.
2. `cross_brand_guarded()` — точная сигнатура и доступность в v2 (impl в `apply_subscription_policy_guards`).
3. Разграничение с §11.15 «brand-confirmation» (это точно Фотон?) — чтобы cross_brand не перехватывал уточнение принадлежности.
