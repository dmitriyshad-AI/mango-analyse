# TZ-116 v2 Offline Modes Report

Дата: 2026-06-15
Ветка: `codex/tz116-offline-understanding`

## Что сделано

- A: добавлен безопасный офлайн-замер CRM LLM vs heuristic: `scripts/run_tz116_crm_llm_offline_measure.py`.
- C: добавлен офлайн-замер каталога вопросов без пересборки основного каталога: `scripts/run_tz116_question_catalog_offline_measure.py`.
- B: в `outcome_linker` добавлен режим `off|shadow|primary`; default `off` сохраняет старый путь. В `shadow/primary` добавлен учёт отрицаний для Tallanto history.
- E: `infer_brand` расширен режимом `legacy|cyrillic_v2`; default `legacy` сохраняет старый путь.
- D: добавлен synthetic/gold evaluator и real mono shadow runner: `scripts/evaluate_tz116_mono_role_assignment.py`, `scripts/run_tz116_mono_role_shadow_real.py`.
- D update: после снятия ПДн-ограничения real mono runner читает `canonical_calls` read-only и готов к `openai_selective` shadow. `primary` заблокирован до gold-набора и регрейда.
- Зафиксирован документ ТЗ-116 v2: `tasks/_inbox_codex/2026-06-15_TZ116v2_gruppa4_offlayn_rezhimy.md`.

## Safety

- AMO/Tallanto/CRM live write: `0`.
- ASR / Resolve+Analyze: `0`.
- Live LLM calls: `0`.
- Реальные ПДн в облако: `0`.
- `stable_runtime` не изменялся.
- Живой путь бота не тронут: `channels/subscription_llm_parts/*`, `insights/sanitizers.py`, `insights/tone_score.py`, `insights/phase2_detectors.py`.

## NEG

- `outcome_linker` default `off` сохраняет legacy-поведение, включая старое ложное срабатывание на `не оплатил`; новый учёт отрицания доступен только в `shadow/primary`.
- CRM offline measure всегда добавляет `offline_measure_no_writeback` и не разрешает writeback.
- Question catalog offline measure не вызывает LLM и не пересобирает основной каталог.
- Mono role evaluator работает только с synthetic/gold ролями и не вызывает OpenAI/ASR.
- Real mono runner открывает SQLite через `mode=ro`, не читает аудио, не запускает ASR, default `off`, `primary` заблокирован без явного флага.
- `infer_brand` default `legacy`; конфликт `Фотон + УНПК` становится `unknown` только в `cyrillic_v2`.

## Тесты

- Точечно: `22 passed`.
- Соседние зоны CRM/question/transcribe: `53 passed, 1 warning`.
- Полный прогон после D-добавки: `3277 passed, 5 skipped, 1 warning in 48.74s`.

## D Real Mono Smoke

- Gold candidate сбор: `50` реальных моно-звонков, `mode=off`, `llm_calls_total=0`, путь: `audits/_inbox/tz116_mono_role_gold_real_20260615/`.
- Shadow smoke без OpenAI key: `10` реальных моно-звонков, `rule_high_conf=9`, `rule_fallback=1`, `llm_calls_total=0`, путь: `audits/_inbox/tz116_mono_role_shadow_real_nokey_smoke_20260615/`.
- Настоящий `openai_selective` shadow не запускался, потому что `OPENAI_API_KEY` отсутствует в текущем окружении и `.env` основной папки.

Warning не связан с изменениями: `urllib3` предупреждает про LibreSSL.

## Смысловая проверка

Verdict: `PASS_WITH_NOTES`.

Проверено, что инструменты создают проверяемую ценность как офлайн-замеры, но не заявляют продуктивное включение `primary`. Остаточные риски: для реального перехода `shadow -> primary` нужны размеченные gold-наборы и отдельное решение по отправке любых реальных текстов в облачные модели.

## Изменённые файлы

- `src/mango_mvp/insights/outcome_linker.py`
- `src/mango_mvp/customer_timeline/canonical_readonly_import.py`
- `scripts/run_tz116_crm_llm_offline_measure.py`
- `scripts/run_tz116_question_catalog_offline_measure.py`
- `scripts/evaluate_tz116_mono_role_assignment.py`
- `scripts/run_tz116_mono_role_shadow_real.py`
- `tests/test_tz116_offline_modes.py`
- `tasks/_inbox_codex/2026-06-15_TZ116v2_gruppa4_offlayn_rezhimy.md`
- `tasks/_done/2026-06-15_TZ116v2_offline_modes_report.md`

## Следующий шаг

Дать Claude #1 этот отчёт и diff по сырью на регрейд. После регрейда можно отдельно решать, какой offline gold-набор собирать для A/C/B/D.
