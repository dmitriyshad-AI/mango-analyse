# TZ-116 Real Codex Measurements Report

Дата: 2026-06-15  
Ветка: `codex/tz116-offline-understanding`

## Что сделано

Проведены настоящие офлайн-замеры по A/C/B/E/D после переключения на Codex CLI. Прямые API-ключи OpenAI не использовались. Записей в AMO, Tallanto, CRM, `stable_runtime` не было.

## Код

- `scripts/build_tz116_crm_fixed_snapshot.py` — фиксированный read-only CRM-набор A через `crm_call.sh`.
- `scripts/run_tz116_be_real_measure.py` — реальные B/E-замеры без модели.
- `scripts/run_tz116_mono_role_gold50_measure.py` — D-замер `codex_selective` на 50 mono-звонках.
- `scripts/run_tz116_question_catalog_offline_measure.py` — C теперь пишет отдельные `rule_vs_gold` и `model_vs_gold`.
- `tests/test_tz116_offline_modes.py` — NEG/регрессии для новых измерителей.

## A: CRM Deal Shadow

Фиксированный набор: `24` закрытые сделки, `12` Фотон + `12` УНПК.  
Read-only RPC ошибок: `0`.

- Сырой snapshot с ПДн: `.codex_local/tz116_crm_fixed_snapshot_20260615_195317/`
- Обезличенный манифест: `audits/_inbox/tz116_crm_fixed_snapshot_20260615_195317/`
- Shadow-замер: `audits/_inbox/tz116_crm_llm_shadow_fixed24_codex_20260615_195654/`
- Codex calls: `24`
- Обработано/сравнено: `24/24`
- `verdict_changed`: `2`
- `risk_changed`: `1`
- `final_writeback_allowed=Да`: `0`

## C: Question Catalog

Полный размеченный набор: `100` вопросов.

- Артефакт: `audits/_inbox/tz116_question_catalog_labeled100_codex_shadow_20260615_192755/`
- Rule vs gold: `37/100`, accuracy `0.37`
- Codex model vs gold: `72/100`, accuracy `0.72`
- Rule/model agree: `30`, disagree: `70`
- Codex calls: `5`
- Основной каталог не пересобирался.

## B: Outcome Linker

Реальные данные без модели.

- Артефакт: `audits/_inbox/tz116_be_real_measure_20260615_192717/`
- Tallanto phone-index: `10671`
- Изменений в Tallanto index: `191`
- Связанных строк `client_chains`: `7902`
- Изменений в связанных строках: `146`
- Крупные flips: `churn_or_refused_after_activity->lost_or_refused = 77`, `won_paid_or_active->known_student_or_lead = 36`, `won_paid_or_active->payment_pending = 15`

Это shadow disagreement / candidate fixes, не gold-accuracy.

## E: Brand Infer

Реальные данные без модели.

- `master_contacts`: legacy known `2642`, cyrillic_v2 known `2429`, delta `-213`, changed `244`
- `amo_contacts`: legacy known `694`, cyrillic_v2 known `1322`, delta `+628`, changed `839`
- `amo_deals`: legacy known `2752`, cyrillic_v2 known `3262`, delta `+510`, changed `620`

Вывод: на AMO-снимках `cyrillic_v2` резко повышает покрытие, но на `master_contacts` он строже и часть строк переводит в `unknown`.

## D: Mono Role Assignment

Прогон на всех `50` gold-кандидатах.

- Артефакт: `audits/_inbox/tz116_mono_role_gold50_codex_measure_20260615_193538/`
- Звонков: `50`
- Фраз: `1723`
- Rule high-confidence: `27`
- Rule low-confidence: `23`
- Codex-called: `23`
- Codex calls: `23`
- Gold-labeled calls: `0`
- Model vs rule: exact `0/23`, mean per-turn agreement `0.5368`

Ограничение: текущий файл `mono_role_gold_review_sample.csv` ещё не содержит ручной `gold_roles`, поэтому точность model/rule против gold честно не посчитана. Сформирована очередь ручной проверки: `mono_role_gold50_manual_review_queue.csv`.

## LLM Calls

- A CRM shadow: `24`
- C question catalog: `5`
- D mono roles: `23`
- B/E: `0`
- Total: `52`

Все вызовы модели шли через Codex CLI. `OPENAI_API_KEY` не использовался.

## Проверки

- Точечные тесты: `15 passed`
- Полный pytest: `3286 passed, 5 skipped, 1 warning`
- Предупреждение: локальный Python использует LibreSSL; на результат тестов и read-only замеры не повлияло.
- `git status` после прогонов содержит только ожидаемые изменения кода/отчёта; raw artifacts игнорируются.

## Смысловой статус

`formal_pass`: да.  
`semantic_pass`: `PASS_WITH_NOTES`.

Что нельзя делать без регрейда:

- включать `primary`;
- писать результаты в AMO/Tallanto/CRM;
- считать D-точность подтверждённой до ручной разметки `gold_roles`;
- трактовать B/E как доказанную точность без gold-проверки.

## Следующие действия

1. Claude #1 регрейдит A/C/B/E/D по сырью.
2. Для D заполнить `gold_roles` хотя бы по очереди low-confidence и повторить тот же скрипт для настоящей точности.
3. По A разобрать 2 расхождения вердикта и 1 расхождение риска.
4. Для B/E выбрать небольшой gold-срез changed rows и подтвердить, какие flips являются реальными исправлениями.
