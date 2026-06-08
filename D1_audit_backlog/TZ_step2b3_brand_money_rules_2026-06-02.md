# ТЗ Шаг 2b.3 — третья волна: денежные бренд-раздельные правила. Для Кодекса. 2026-06-02.

Автор: Клод 1 (read-only). Реализует Кодекс. `.phase12`, заморозка, без пилота/stable_runtime.
Основание: 2b.1 (`7b28fa28`) + 2b.2 (`3a539a5a`) приняты. MIGRATED=8.

## Почему узкая волна (2 правила)
2b.3 = **installment + discount**. Это ПЕРВЫЙ заход cross-brand риска в миграцию — главное правило проекта
(«бренды не смешиваются»). Не разбавляю его форматом/ценами (они — динамические KB-значения + формат-гард,
отдельная волна 2b.4). Цель волны: доказать, что бренд-разделение в правилах держится так же строго, как в
монолите. price + format_choice → 2b.4; trial/schedule/camp/enrollment → 2b.5.

## Материалы
`rules_registry.yaml` (installment, discount), `REVIEW_step2b1/2b2_*`, контракты installment(ЭТАЛОН)/discount,
память: рассрочка Фотон 6/10/12 + Долями; УНПК рассрочки НЕТ (вместо неё семестр 10% / год 14%, оба
client-safe).

## Анкеры (прочитать функции + константы)
- `_installment_safe_template`:4491, `_discount_safe_template`:4399.
- Константы: `FOTON_INSTALLMENT_SAFE_TEXT`:127, `UNPK_INSTALLMENT_APPROVED_FALLBACK_TEXT`:182,
  `FOTON_SECOND_SUBJECT_DISCOUNT_TEXT`:107, `UNPK_SECOND_SUBJECT_DISCOUNT_TEXT`:111,
  `MULTICHILD_DISCOUNT_TEXT`:121, `DISCOUNT_STACKING_SAFE_TEXT`:126.
- Интенты уже есть: `conversation_intent_plan.py` discount:303, installment:304, payment_by_invoice_monthly:282.
  ВАЖНО: installment в сигналах алиасится в `payment_method` (intent-plan:137) — учесть оба пути.
- **Страховка гейта уже в коде** (не дублировать в правиле): `_produce_cross_brand_template`:808,
  cross_brand safe_template:890, `guard_promocode_leak`:1928, `_cross_brand_safe_template`:2643.

## Правила переноса

### installment (КРИТИЧНО cross-brand)
```yaml
data:
  foton: {has_installment: true, months: [6,10,12], dolyami: true}
  unpk: {has_installment: false, semester_pct: 10, year_pct: 14}
```
- Фотон: рассрочка 6/10/12 мес + Долями (из факта); 6/12 — через менеджера.
- **УНПК: рассрочки НЕТ.** Правило ОБЯЗАНО ответить честно (не молчать, не выдумать рассрочку УНПК):
  «рассрочки нет, но есть скидка за оплату: семестр 10%, год 14%» (из `UNPK_INSTALLMENT_APPROVED_FALLBACK_TEXT`).
- blocking → гейт: `cross_brand` (не сравнивать Фотон/УНПК), `brand_leak`.

### discount (бренд+формат значения)
```yaml
data:
  second_subject: {foton_online_pct: 30, foton_offline_pct: 20, unpk_pct: 20}
  mfti_staff_pct: 10
  multichild_pct: 10           # по СТАТУСУ семьи, не по числу детей в CRM
  stacking: take_max           # скидки НЕ суммируются — наибольшая
```
- каждая скидка — с условием применения.
- blocking → гейт: `promocode_leak` (промокоды убраны из клиентского слоя), `cross_brand`.

## Общие инварианты (как 2b.1/2b.2)
- значения процентов/месяцев — из rule data/факта (инвариантные бизнес-правила); НЕ выдумывать.
- route=bot_answer_self; blocking_conditions исполняет ВЫХОДНОЙ ГЕЙТ (cross_brand/promocode/brand_leak уже в
  коде) — правило их только декларирует.
- **brand_split строго:** в одном клиентском ответе НИКОГДА не упоминать оба бренда и не сравнивать их условия.
- диспетчер — та же точка; фолбэк 6 оставшихся правил цел; текст-константы не удалять (кандидат на 2b-чистку).
- НЕ расширять `_manager_route_migrated_rules_override_allowed` (из 2b.2) на installment/discount без явных
  проверок P0 — для этих двух override manager_only НЕ нужен.

## Четыре правила Кодекса
1. Read-only чтение 2 функций + подтверждение, что УНПК-ветка installment резолвит fallback-факт (иначе
   правило молчит на УНПК-рассрочке — это была бы регрессия). 2. Простота: переиспользовать интенты. 3.
   Хирургия: только маршрут installment/discount + rules_engine; 6 правил не трогать. 4. Цель → проверяемый
   критерий: на каждую ветку + каждое cross-brand/promocode/stacking условие — негатив-тест.

## Тест выхода
1. **Офлайн-регрессия** на 58 диалогах Прогона 1: 6 немигрированных правил — дифф 0.
2. **НЕГАТИВ-тесты (обязательны):**
   - **УНПК «есть рассрочка?»** → честно НЕТ + семестр 10%/год 14%; НЕ выдумать рассрочку УНПК; НЕ упомянуть
     Фотон.
   - **cross-brand** (в УНПК-боте «а в Фотоне рассрочка есть?») → НЕ сравнивать, фраза про отдельные
     организации; в одном ответе только один бренд.
   - **stacking** «многодетная + второй предмет, сложите?» → наибольшая, НЕ сумма.
   - **multichild** → по статусу семьи, не по числу детей.
   - **promocode** «есть промокод?» → НЕ выдать (убраны из клиентского слоя).
   - **second subject** Фотон онлайн 30 / очно 20, УНПК 20 — бренд+формат верный, без cross-brand.
   - **P0-override** (долг-дисциплина): installment/discount НЕ перебивают manager_only/P0.
3. **Полный pytest** зелёный (те же ~9 инфра-фейлов).
4. **Один точечный прогон** по installment/discount × оба бренда + cross-brand провокации (~15-18 диалогов,
   `--parallel 4`): убедиться, что УНПК-рассрочка отвечается честно, бренды не смешиваются, промокодов нет.

## Что НЕ трогать
6 немигрированных (price, format_choice, trial, schedule, camp_lvsh, enrollment_process), гейт Шага 1,
cross_brand/promocode-гарды, stable_runtime, KB-снимок. Не удалять константы. Не git reset/checkout/clean.

## Отчёт Кодекса
Файлы/строки и почему; дифф регрессии (0 на немигрированных); NEG-результаты (особо: УНПК-рассрочка честно +
cross-brand + stacking); ревизия коммита для моей проверки; кандидаты 2b.4 (price+format_choice).

Я (Клод) проверю по сырью + адверсариально (cross-brand смешение, выдумка рассрочки УНПК, суммирование скидок,
промокод, P0-override) и отрегрейжу точечный прогон перед 2b.4.
