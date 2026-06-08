# Phase 12 — дизайн диспетчера precedence safe-шаблонов в v2

Автор: второй Claude. Дата: 2026-05-29. Снимок: HEAD `36e23cb8`. Read-only.
Назначение: единый механизм выбора ОДНОГО safe-шаблона в v2-цепочке вместо хрупких ручных `or metadata.get("..._applied")` цепочек legacy. На него ссылаются все 9 ТЗ Блока А.

## Проблема (из аудита)

В legacy (`apply_subscription_policy_guards`, ~`subscription_llm.py:2120-2750`) каждый шаблон гейтится растущей цепочкой `"" if cross_brand_guarded() or metadata.get("terminal_safe_template_applied") or ... else _X_safe_template(...)` (примеры `:2370/2378/2386/2401/2416/2453`). Precedence = «первый сработавший ставит флаг, остальные пропускают себя». Минусы: добавление шаблона требует ручного апдейта skip-условий ВСЕХ последующих; забыли — двойная перезапись `draft_text`. На 9 P0 это не масштабируется.

## Контракт диспетчера

Вход: `result: SubscriptionDraftResult`, `client_message`, `context`. Выход: тот же `result` с применённым максимум ОДНИМ шаблоном (или без изменений).

Каждый шаблон описывается записью:
```python
@dataclass(frozen=True)
class TemplateSpec:
    name: str                 # "cross_brand", "olympiad_online", ...
    priority: int             # меньше = выше приоритет
    produce: Callable[[SubscriptionDraftResult, str, Mapping], str]  # = существующий _X_safe_template
    route_on_apply: str       # "manager_only" | "draft_for_manager" | "keep_or_draft"
    flag: str                 # "X_safe_template_applied"
    checklist: str
REGISTRY: tuple[TemplateSpec, ...] = (...)  # отсортирован по priority
```

## 3 варианта реализации

### Вариант 1 — «монотонная композиция первого срабатывания по приоритету» (РЕКОМЕНДУЮ)

Идём по REGISTRY в порядке priority; первый `produce()`, вернувший непустой текст, применяется, остальные пропускаются. Заменяет ручные skip-цепочки одной петлёй.
```python
def apply_template_dispatcher(result, *, client_message, context):
    if cross_brand_guarded(result, context):           # глобальный гейт, как сейчас
        spec = _CROSS_BRAND_SPEC
        text = spec.produce(result, client_message, context)
        return _apply(result, spec, text) if text else result
    for spec in REGISTRY:                              # уже по priority
        text = spec.produce(result, client_message, context)
        if text:
            return _apply(result, spec, text)          # один шаблон, точка
    return result
```
- **Плюсы:** один источник порядка (priority), добавление шаблона = одна строка в REGISTRY; невозможна двойная перезапись (выходим на первом).
- **Минусы:** «первый по приоритету» может перекрыть более релевантный, если priority задан неточно → приоритеты нужно выверить (см. таблицу ниже).

### Вариант 2 — «приоритет по типу claim» (классификация → один шаблон)

Сначала классифицировать тип запроса (brand / identity / guarantee / olympiad / price / camp / ...), затем вызвать ровно соответствующий шаблон. Маппинг intent→template.
- **Плюсы:** выбор по смыслу, не по порядку; ближе к v2-философии (намерение).
- **Минусы:** нужен надёжный классификатор типа (это `conversation_intent_plan.primary_intent`/`fact_scope` — уже есть, переиспользовать); риск «нет ветки под редкий тип» → fallthrough.

### Вариант 3 — «теги-кандидаты + разрешение конфликта» (для отладки)

Каждый шаблон возвращает (text, claimed_topics); собрать всех кандидатов, затем разрешить по priority + залогировать конфликт (≥2 кандидата). Применяется один, остальные пишутся в metadata `template_conflicts` для аудита.
- **Плюсы:** видимость конфликтов (для тестов/мониторинга).
- **Минусы:** дороже (зовём все produce()); для прода избыточно.

**Решение:** Вариант 1 для прода + диагностику Варианта 3 включать в тестах (собрать всех кандидатов, ассертить, что претендент ≤1 на ход; ≥2 → разобрать приоритет).

## Точка вставки в v2

В `_apply_dialogue_contract_v2_guard_chain` (`:994-1029`): добавить `apply_template_dispatcher` ПОСЛЕ контентных verifier-guard'ов (после `apply_unconfirmed_operational_specificity_guard` `:1023`) и ПЕРЕД `apply_funnel_policy_guard` (`:1027`) / `_dialogue_contract_v2_route_permission_guard` (`:1028`) / `_sanitize_dialogue_contract_client_text` (`:1029`). Затем `_reverify_dialogue_contract_text_change` (как для остальных шагов). Sanitize остаётся последним (чтобы вычистить 3-е лицо/утечки и из шаблонного текста — см. 11.12).

## Приоритеты (priority, меньше=выше) для 9 P0 + контекст

| priority | template | почему так |
|---|---|---|
| 10 | cross_brand | брэнд-leak перекрывает всё |
| 20 | terminal-identity | природа бота — раньше контента |
| 30 | result_guarantee | обещание балла — раньше цен |
| 31 | admission_guarantee | обещание поступления |
| 40 | matkap-guarantee | финансовое обещание |
| 41 | tax-guarantee | финансовое обещание |
| 50 | olympiad_online | scope продукта раньше общей цены |
| 51 | offline_grade_scope (новый) | scope класса раньше цены |
| 52 | closed_world (новый) | «нет других» раньше тематич. ответа |
| (P1) | camp/format/pricing/installment/discount/trial/... | ниже, по теме |

## Конфликт ценовых шаблонов с v2-guard `unsupported_promise`

`unsupported_promise` (`apply_unsupported_promise_guard:1019` в v2) и ценовые шаблоны (`result_guarantee/pricing/discount`) оба трогают числа/обещания. Развязка:
- **Порядок:** `unsupported_promise` (verifier, `:1019`) идёт ДО диспетчера шаблонов (после `:1023`). Verifier работает с ЧИСЛАМИ в драфте (помечает/режет неподтверждённые); шаблоны — это ЗАМЕНА текста на безопасный.
- **Правило непересечения:** если `unsupported_promise_detected` уже сработал и увёл в `manager_only` — диспетчер шаблонов НЕ перезаписывает текст (шаблон применяется только если route ещё допускает ответ). Добавить в `_apply`: `if "unsupported_promise_detected" in result.safety_flags and spec.name in PRICE_CLUSTER: skip`.
- **result_guarantee — исключение:** это шаблон-ОТКАЗ от обещания (route→draft_for_manager + placeholder), он ДОЛЖЕН срабатывать даже поверх unsupported_promise (усиливает безопасность, не ослабляет). Поэтому result/admission_guarantee имеют высокий priority (30/31) и не блокируются price-cluster-правилом.
- Итог: `unsupported_promise` режет числа (защита), шаблон-гарантия отказывает в обещании (защита), ценовые шаблоны (pricing/discount) НЕ применяются, если число уже срезано verifier'ом (избегаем двойной обработки).

## Тесты на конфликты

1. **Два претендента:** вход, под который подходят и `cross_brand`, и `terminal-identity` → применяется cross_brand (priority 10); ассерт ровно один `*_applied` флаг.
2. **olympiad vs pricing:** «олимпиадная физика 11, сколько?» → olympiad_online (50) раньше pricing → не путать продукт.
3. **result_guarantee поверх unsupported_promise:** «гарантируете 100 баллов?» + число в драфте → result_guarantee применяется, route draft_for_manager, не остаётся «100 баллов».
4. **price-cluster под срезанным числом:** unsupported_promise уже сработал → pricing-шаблон НЕ перезаписывает (skip), нет двойной обработки.
5. **Идемпотентность:** повторный прогон диспетчера = тот же результат (выходим на первом совпадении).
6. **Нет претендентов:** обычный ответ из фактов → диспетчер не трогает текст.

## Открытые вопросы для Кодекса

1. `cross_brand_guarded()` — текущая сигнатура/доступность в v2-контексте (в legacy это helper в `apply_subscription_policy_guards`). Подтвердить, что доступен в v2-цепочке.
2. Точные имена produce-функций ценового кластера (`_pricing_safe_template` — def-строку не зафиксировал; `_installment/_discount/_price_installment_multitopic` подтверждены).
3. Где сейчас выставляется `skip_green_template_overwrite` и нужен ли он в v2 (в legacy завязан на answer_contract).
4. Должен ли диспетчер в v2 нормализовать `topic` (legacy делает `program_topic_normalized` и т.п.) — или topic в v2 не используется downstream.
