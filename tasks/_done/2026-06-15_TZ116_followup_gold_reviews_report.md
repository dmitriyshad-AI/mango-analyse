# TZ-116 Follow-up Gold Reviews

Дата: 2026-06-15  
Ветка: `codex/tz116-offline-understanding`

## Рамка

Выполнены follow-up проверки после `PASS_WITH_NOTES`:

- C: 8 кейсов, где модель сломала верное правило.
- B: gold-срез outcome flips.
- E: gold-срез `master_contacts`, ушедших в `unknown`.
- D: ручная разметка 23 low-confidence mono-звонков и пересчёт тем же скриптом.

Primary/writeback нигде не включались. Записей в AMO/Tallanto/CRM/DB не было.

## Артефакты

- C/B/E gold reviews: `audits/_inbox/tz116_followup_gold_reviews_20260615/`
- D ручная разметка: `audits/_inbox/tz116_mono_role_gold23_manual_20260615/`
- D rerun: `audits/_inbox/tz116_mono_role_gold23_rerun_20260615_221929/`

## C: Model Broke Correct Rule

Найдено `8` регрессий: `calib_004`, `calib_011`, `calib_035`, `calib_051`, `calib_053`, `calib_060`, `calib_078`, `calib_080`.

Классы ошибок:

- короткие/обрывочные формулировки модель слишком легко переводит в `unclear` или `non_question`;
- денежные темы смешиваются: цена, скидка, срок действия цены, график оплаты, статус платежа;
- негативная окраска не должна автоматически становиться жалобой.

Решение перед любым включением C-primary: гибридный guard. Если правило уверенно попало в эти классы, модель не должна перетирать его без дополнительного условия.

## B: Outcome Flips Gold Slice

Срез: `25` flips.

Итого:

- `true_fix`: `13`
- `false_fix`: `6`
- `unclear`: `6`

По классам:

- `won_paid_or_active -> known_student_or_lead`: `8 true_fix`, `1 false_fix`, `1 unclear`
- `won_paid_or_active -> payment_pending`: `5 true_fix`, `5 false_fix`, `5 unclear`

Вывод:

- `known_student_or_lead` в основном полезно снимает ложный `won`, но часто это консервативный fallback вместо более точного `lost/refused/churn`.
- `payment_pending` слабый: много ложных срабатываний на слова про чек, долг, счёт и устаревшие состояния. В primary его включать нельзя.

## E: Brand Loss Gold Slice

Пул `master_contacts`: `228` строк, из них `167 unpk->unknown`, `61 foton->unknown`.

Срез: `20` строк.

Итого:

- `expected_fail_closed`: `8`
- `false_negative`: `10`
- `unclear`: `2`

По классам:

- `unpk->unknown`: `6 expected_fail_closed`, `1 false_negative`, `2 unclear`
- `foton->unknown`: `9 false_negative`, `2 expected_fail_closed`

Вывод:

- `unpk->unknown` чаще выглядит правильным fail-closed из-за смешения брендов в одной строке.
- `foton->unknown` часто теряет реальные совпадения: падежи, множественное число и склеенные формы вроде `ЦДПФОТОН`.
- Перед включением `cyrillic_v2` как основной логики нужно доработать Foton-распознавание, не ослабляя cross-brand fail-closed.

## D: Mono Role Gold-23

Ручная разметка: `23` low-confidence/Codex cases, `924` реплики.  
Частично неуверенные ручные кейсы: `4`.

Пересчёт тем же скриптом:

- `rule_vs_gold`: exact `0/23`, mean per-turn `0.5377`
- `model_vs_gold`: exact `3/23`, mean per-turn `0.9489`
- `selected_vs_gold`: exact `3/23`, mean per-turn `0.9489`
- `model_vs_rule`: exact `0/23`, mean per-turn `0.5388`
- Codex calls in rerun: `23`

Вывод:

- На low-confidence зоне модель резко лучше правила по репликам.
- Exact-call metric слишком строгая для длинных звонков: одна ошибка в массиве делает весь звонок incorrect.
- Для D-primary всё равно нужен регрейд; безопаснее использовать per-turn threshold + spot review.

## LLM Calls

Новые вызовы модели в этом follow-up:

- D rerun: `23`
- C/B/E follow-up: `0`

Все модельные вызовы через Codex CLI, без `OPENAI_API_KEY`.

## Semantic Status

`formal_pass`: да, отчёты и артефакты собраны.  
`semantic_pass`: `PASS_WITH_NOTES`.

Ограничения:

- C подтверждён как победа модели в целом (`72/100` vs `37/100`), но 8 регрессий надо закрыть guard-логикой.
- B `payment_pending` не готов к primary.
- E `cyrillic_v2` нельзя включать без доработки Foton-форм.
- D полезен в low-confidence зоне, но требует регрейда и аккуратной метрики.

## Следующие безопасные действия

1. C: добавить guard/regression tests на 8 кейсов.
2. B: усилить `payment_pending`: требовать более свежий/явный платежный сигнал и не путать долг/чек/счёт со статусом pending.
3. E: расширить Foton regex на падежи/склейки, сохранив fail-closed для смешанных брендов.
4. D: добавить progress/resume для длинных Codex role runs и оценивать per-turn метрику как основную.
