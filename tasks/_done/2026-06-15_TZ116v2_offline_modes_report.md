# TZ-116 v2 Offline Modes Report

Дата: 2026-06-15
Ветка: `codex/tz116-offline-understanding`

## Что сделано

- A: добавлен безопасный офлайн-замер CRM LLM vs heuristic: `scripts/run_tz116_crm_llm_offline_measure.py`.
- C: добавлен офлайн-замер каталога вопросов без пересборки основного каталога: `scripts/run_tz116_question_catalog_offline_measure.py`.
- B: в `outcome_linker` добавлен режим `off|shadow|primary`; default `off` сохраняет старый путь. В `shadow/primary` добавлен учёт отрицаний для Tallanto history.
- E: `infer_brand` расширен режимом `legacy|cyrillic_v2`; default `legacy` сохраняет старый путь.
- D: добавлен synthetic/gold evaluator и real mono shadow runner: `scripts/evaluate_tz116_mono_role_assignment.py`, `scripts/run_tz116_mono_role_shadow_real.py`.
- D update: `openai_selective` для TZ-116 не применяется. Добавлен `codex_selective`: назначение ролей через Codex CLI с изолированным `CODEX_HOME`, нейтральной персоной, без `OPENAI_API_KEY`. `primary` в runner заблокирован до gold-набора и регрейда.
- A/C update: добавлен явный `--llm-source codex` для shadow-замеров через Codex CLI. Старые `off/precomputed` режимы сохранены, основной каталог не пересобирается.
- Deal-aware Codex runtime больше не наследует пользовательские `service_tier`/`personality` из `~/.codex/config.toml`; создаётся нейтральный runtime config.
- Зафиксирован документ ТЗ-116 v2: `tasks/_inbox_codex/2026-06-15_TZ116v2_gruppa4_offlayn_rezhimy.md`.

## Safety

- AMO/Tallanto/CRM live write: `0`.
- ASR / Resolve+Analyze: `0`.
- Прямые OpenAI API calls: `0`.
- `OPENAI_API_KEY`: не заводился и вычищается из env для новых Codex CLI контуров.
- Codex CLI shadow calls: `2` (`C=1`, `D=1`, `A=0`).
- `stable_runtime` не изменялся.
- Живой путь бота не тронут: `channels/subscription_llm_parts/*`, `insights/sanitizers.py`, `insights/tone_score.py`, `insights/phase2_detectors.py`.

## NEG

- `outcome_linker` default `off` сохраняет legacy-поведение, включая старое ложное срабатывание на `не оплатил`; новый учёт отрицания доступен только в `shadow/primary`.
- CRM offline measure всегда добавляет `offline_measure_no_writeback` и не разрешает writeback; Codex-режим требует dossier во входе.
- Question catalog offline measure вызывает модель только при `--llm-source codex`; основной каталог не пересобирает.
- Mono role evaluator работает только с synthetic/gold ролями и не вызывает OpenAI/ASR.
- Real mono runner открывает SQLite через `mode=ro`, не читает аудио, не запускает ASR, default `off`, `primary` заблокирован без явного флага; shadow использует `codex_selective`, не `openai_selective`.
- `infer_brand` default `legacy`; конфликт `Фотон + УНПК` становится `unknown` только в `cyrillic_v2`.
- Нейтральный Codex runtime не копирует `service_tier`/`personality` и не передаёт `OPENAI_API_KEY`.

## Тесты

- Расширенная точечная регрессия: `61 passed, 1 warning`.
- Полный прогон после Codex CLI правок: `3283 passed, 5 skipped, 1 warning in 42.55s`.

Warning не связан с изменениями: `urllib3` предупреждает про LibreSSL.

## A CRM Shadow

- Кодовый режим `--llm-source codex` реализован и покрыт тестом.
- Реальный A shadow не запускался: в текущем worktree и основной папке не найден фиксированный offline-набор с `dossier` для deal-aware анализа. Запуск без dossier был бы фиктивным.
- При наличии входа формат ожидается как JSONL: `case_id`, `heuristic_analysis`, `dossier`; запись в AMO/CRM всё равно заблокирована `offline_measure_no_writeback`.

## C Question Catalog Shadow

- Реальный малый shadow: `5` клиентских вопросов из `product_data/telegram_dynamic_test_sets/real_questions_20260531.jsonl`.
- Codex CLI calls: `1`.
- Сравнение Codex ↔ rule: `agree=2`, `disagree=3`.
- Gold labels в этом наборе отсутствуют: `gold_labeled_total=0`, поэтому accuracy не считается.
- Путь: `audits/_inbox/tz116_question_catalog_codex_real_20260615/`.

## D Real Mono Smoke

- Gold candidate сбор: `50` реальных моно-звонков, `mode=off`, `llm_calls_total=0`, путь: `audits/_inbox/tz116_mono_role_gold_real_20260615/`.
- Codex shadow smoke: `10` реальных моно-звонков, `rule_high_conf=9`, `codex_cli=1`, `llm_calls_total=1`, путь: `audits/_inbox/tz116_mono_role_shadow_real_codex_smoke2_20260615/`.
- Первый пробный Codex shadow поймал инфраструктурную проблему `service_tier`; исправлено нейтральным runtime config и закреплено тестом.

## Смысловая проверка

Verdict: `PASS_WITH_NOTES`.

Проверено, что инструменты создают проверяемую ценность как офлайн-замеры и не заявляют включение `primary`. Остаточные риски: A требует настоящий dossier-набор; C требует gold labels для accuracy; D требует ручную проверку gold-ролей перед любым primary.

## Изменённые файлы

- `src/mango_mvp/amocrm_runtime/deal_llm.py`
- `src/mango_mvp/services/transcribe.py`
- `src/mango_mvp/insights/outcome_linker.py`
- `src/mango_mvp/customer_timeline/canonical_readonly_import.py`
- `scripts/run_tz116_crm_llm_offline_measure.py`
- `scripts/run_tz116_question_catalog_offline_measure.py`
- `scripts/evaluate_tz116_mono_role_assignment.py`
- `scripts/run_tz116_mono_role_shadow_real.py`
- `tests/test_amocrm_deals.py`
- `tests/test_codex_merge.py`
- `tests/test_dialogue_format.py`
- `tests/test_tz116_offline_modes.py`
- `tasks/_inbox_codex/2026-06-15_TZ116v2_gruppa4_offlayn_rezhimy.md`
- `tasks/_done/2026-06-15_TZ116v2_offline_modes_report.md`

## Следующий шаг

Дать Claude #1 этот отчёт и diff по сырью на регрейд. После регрейда отдельно решить: где взять A dossier-набор, как разметить C gold labels и какие D gold-строки проверять вручную.
