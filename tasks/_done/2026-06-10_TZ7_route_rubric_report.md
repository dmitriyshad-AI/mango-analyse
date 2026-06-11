# TZ-7 route rubric report

Дата: 2026-06-10

Кодовый коммит: `411984f6` (`Add direct path route rubric`)

## Что сделано

- Добавлен флаг `TELEGRAM_ROUTE_RUBRIC`, default OFF.
- Флаг не добавлен в профиль `pilot_gold_v1`.
- В прямой путь добавлена рубрика выбора маршрута в prompt только при включенном флаге.
- Строка миссии про «менеджер свяжется» под тем же флагом скоупится только на черновик для менеджера.
- Добавлена одна перегенерация строго после успешного direct draft-вызова и до semantic verifier/gate, только для `draft_for_manager` без `missing_facts` при наличии фактов.
- Перегенерация не меняет `direct_path_regenerated`; отдельные поля: `rubric_enabled`, `rubric_regenerated`, `rubric_reason`.
- Добавлен annotate-only индикатор `deferral_text_in_self`.
- В summary раннера добавлен блок `direct_path_rubric`.

## NEG / контроль

- `pilot_gold_v1` не включает `TELEGRAM_ROUTE_RUBRIC`.
- Golden prompt OFF совпадает с прежней структурой, ON добавляет рубрику и скоупит строку миссии.
- Матрица регенерации: self-route, `missing_facts`, отсутствие фактов, OFF и preblock не вызывают повтор.
- Ошибка второго вызова оставляет первый draft и пишет `regen_failed:*`.
- Перегенерированный self-текст всё равно проходит authoritative gate; неподтверждённое продуктовое число блокируется.
- Анти-Волна-5: рубрика не повышает route кодом при OFF/без успешной перегенерации.

## Проверки

- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_subscription_llm_draft_provider.py tests/test_telegram_dynamic_client_sim.py`
  - `517 passed`
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests`
  - `2893 passed, 2 skipped, 1 warning`

## Кандидаты для будущего смоука

- Тревожный родитель: модель должна не уходить в draft без `missing_facts`, если факты покрывают вопрос.
- «За два предмета выйдет X»: проверка, что route-рубрика не разблокирует продуктовые числа без факта.
- Вопрос про порядок записи/оформления: рассказать процесс по факту, но действие записи оставить draft.
- Вопрос с реальным отсутствующим процессным фактом: `draft_for_manager` должен содержать конкретные `missing_facts`.

Симулятор и М1-прогоны не запускались: по ТЗ замер позже отдельной задачей.

## Ночной локальный замер 2026-06-10

Контекст:
- Дерево: `2d2da7da`.
- Машина: главный мак, M1 не трогался.
- Временный `CODEX_HOME`: `/private/tmp/mango_codex_home_rubric_20260610_fast`.
- Временный config содержит `service_tier = "fast"`; основной `~/.codex/config.toml` не менялся.
- Live AMO-проба не запускалась.
- Набор: `/Users/dmitrijfabarisov/Yandex.Disk.localized/OpenClaw/Actual Mango Tests/smoke_v2_acceptance_2026-06-08.jsonl`.
- Команда: `scripts/run_telegram_dynamic_client_sim.py --parallel 4 --judge-prompt-version v9`.
- Сравнение только внутри пары OFF/ON на этой машине; с M1 не сравнивалось.

OFF:
- Out-dir: `runs/20260610_rubric_base_smoke89`.
- Env: `TELEGRAM_DIRECT_PATH_PILOT_CONFIG=pilot_gold_v1`.
- Итог: `89 dialogs`, `26 PASS`, `57 PASS_WITH_NOTES`, `6 FAIL`, `ok=true`.
- `config_validity.invalid=false`.
- `bot_direct_draft=300`.
- `bot_semantic_output_verifier=390`.
- `bot_faithfulness=0`.
- `direct_path_rubric.rubric_enabled=0`.
- FAIL: `sm_f_xbrand2`, `sm_u_camp1`, `sm_u_camp_zvsh`, `sm_u_discount_year`, `sm_u_waitlist`, `sm_f_summer_prog`.

ON:
- Out-dir: `runs/20260610_rubric_on_smoke89`.
- Env: `TELEGRAM_DIRECT_PATH_PILOT_CONFIG=pilot_gold_v1`, `TELEGRAM_ROUTE_RUBRIC=1`.
- Итог: `89 dialogs`, `34 PASS`, `47 PASS_WITH_NOTES`, `8 FAIL`, `ok=true`.
- `config_validity.invalid=false`.
- `bot_direct_draft=295`.
- `bot_semantic_output_verifier=362`.
- `bot_faithfulness=0`.
- `direct_path_rubric.rubric_enabled=321`.
- FAIL: `sm_f_camp1`, `sm_u_camp1`, `sm_u_install`, `sm_u_discount_year`, `sm_u_price_12`, `sm_f_pick2kl`, `sm_u_mixed`, `sm_f_format_both`.

Валидность замера:
- Оба плеча завершились полностью.
- В обоих плечах direct draft и semantic verifier реально работали.
- В ON рубрика реально включалась (`rubric_enabled=321`).
- `config_validity.invalid=false` в обоих плечах.
