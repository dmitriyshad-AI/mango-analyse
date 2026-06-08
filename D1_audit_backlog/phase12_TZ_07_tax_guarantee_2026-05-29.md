# Phase 12 · TZ-07 — миграция `tax`-guarantee в v2

Снимок: HEAD `36e23cb8`. Приоритет диспетчера: 41. Зависит от: диспетчера.

## 1. Цель
По налоговому вычету не гарантировать возврат от ФНС; не раскрывать юр.детали лицензий клиенту; отвечать утверждёнными формулировками (ФНС рассматривает; лицензия есть; справка помогает). Финансово-юр. обещание + юр.данные = риск.

## 2. Текущее состояние в legacy
- Селектор: `_tax_safe_template(result, *, client_message, context)` — `subscription_llm.py:5421`. Применение: `:2378-2384` — flag `tax_safe_template_applied`, checklist «Налоговый вычет: не гарантировать возврат от ФНС».
- Тексты: `TAX_ONLINE_FORM_SAFE_TEXT:199`, `TAX_FNS_REVIEW_SAFE_TEXT:203`, `TAX_AMOUNT_SAFE_TEXT:204`, `TAX_LICENSE_SAFE_TEXT:211`.
- **В v2 НЕ вызывается.**
- CLAUDE.md: налоговый вычет можно объяснять справочно (с active_brand); клиентская формулировка про лицензию — «У нас есть лицензия на образовательную деятельность» (без номеров/дат). P0 — guarantee-блок ФНС + не-раскрытие юр.номеров; справка — P1.

## 3. Точка вставки в v2
Диспетчер, priority 41 (рядом с matkap). Skip как legacy (`:2378`).

## 4. Зависимости (KB)
Тексты — константы; `TAX_AMOUNT_SAFE_TEXT` содержит суммы лимитов — числа в whitelist `_is_verified_safe_numeric_template:1631`. Лицензионные client-safe формулировки уже в v6.3 (`licenses_client_safe_summary` — видел в транскриптах). **Новых фактов не нужно.** Блок Г: OK. ⚠️ Контроль: юр.номера лицензий (внутренние по CLAUDE.md) НЕ должны попадать в клиентский текст — `allowed_client_text_has_no_license_numbers` гейт KB это держит; шаблон отдаёт только «есть лицензия».

## 5. Точная правка
```python
_TAX_SPEC = TemplateSpec(
    name="tax", priority=41,
    produce=lambda r, cm, ctx: _tax_safe_template(r, client_message=cm, context=ctx),
    route_on_apply="keep_or_draft",
    flag="tax_safe_template_applied",
    checklist="Налоговый вычет: не гарантировать возврат ФНС; лицензия без номеров.",
)
```

## 6. Регрессы
Positive:
1. `«вернут ли мне налоговый вычет?»` → TAX_FNS_REVIEW_SAFE_TEXT («ФНС рассматривает, не обещаем»).
2. `«сколько вернут?»` → TAX_AMOUNT_SAFE_TEXT (лимиты из факта, без обещания одобрения).
3. `«у вас есть лицензия?»` → TAX_LICENSE_SAFE_TEXT («есть лицензия…»), без номеров.
4. `«как оформить вычет?»` → TAX_ONLINE_FORM_SAFE_TEXT.
Контрольные negative:
5. `«дайте справку для вычета»` (справочно) → справочный ответ/менеджер, не жёсткий отказ; НЕ high_risk (находка 11.2).
6. Юр.номер лицензии НЕ появляется в клиентском тексте (контроль — `allowed_client_text_has_no_license_numbers`).
7. Числа лимитов вычета → не ловятся unsupported_promise (whitelist).

## 7. Backward compatibility
- Legacy tax тесты зелёные.
- KB-гейт `allowed_client_text_has_no_license_numbers` не нарушается.
- Справочная часть вычета (CLAUDE.md) не зарезается; 11.2 (high_risk на «оригинал договора/справка для вычета») — не ужесточать обратно.

## 8. Открытые вопросы Кодексу
1. P0 (guarantee ФНС + юр.номера) vs P1 (справка) — мигрировать весь `_tax_safe_template` (рекомендую) или только guarantee.
2. `TAX_AMOUNT_SAFE_TEXT` числа → whitelist `_is_verified_safe_numeric_template`.
