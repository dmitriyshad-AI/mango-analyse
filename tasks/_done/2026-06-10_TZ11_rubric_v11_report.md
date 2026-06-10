# TZ-11 rubric v1.1 and next: report

Дата: 2026-06-10

## Итог

- Часть 1 выполнена коммитом `cad95971` (`Add route rubric v1.1 derivative guidance`).
- Часть 2 выполнена коммитом `33d9ad94` (`Classify derived product numbers in output gate`).
- Часть 3 выполнена коммитом `26d7bcd3` (`Enable route rubric in pilot profile and add smoke18`).
- Дополнительная тестовая стабилизация: `bc6d88a8` (`Stabilize deal attribution confidence test`).
- Текущий HEAD: `bc6d88a8`.
- `TELEGRAM_ROUTE_RUBRIC` включен в профиль `pilot_gold_v1`.
- Smoke18 прогнан на главном маке; разбор содержательных результатов оставлен архитектору.

## Часть 1

Изменения:
- `src/mango_mvp/channels/subscription_llm.py`: в `DIRECT_PATH_ROUTE_RUBRIC_BLOCK` добавлено правило v1.1:
  не вычислять новые проценты/скидки/суммы/итоги из цен, не подтверждать расчеты клиента, избегать сравнительных оценок без факта.
- `tests/test_subscription_llm_draft_provider.py`: переснят ON golden для route rubric prompt.

Проверки:
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_subscription_llm_draft_provider.py -k route_rubric`
  - `7 passed, 423 deselected`
- Полный pytest после части 1:
  - `1 failed, 2894 passed, 2 skipped`
  - Единственный FAIL: `tests/test_deal_aware_stage2_attribution.py::test_confidence_thresholds_post_recalibration`.
  - Причина по сырью: ожидается `high`, фактически `medium` на старом `deal_updated_at=2026-05-10T10:00:00+00:00`; это дата-зависимый AMO attribution тест, не связанный с TZ-11.

## Часть 2

Изменения:
- `src/mango_mvp/channels/subscription_llm.py`:
  - добавлен код findings `derived_product_number`;
  - политика `GATE_BLOCKING_CODES`: `derived_product_number -> downgrade_keep_text`;
  - код не добавлен в `DIRECT_PATH_REPLACE_TEXT_GATE_CODES`, текст черновика сохраняется менеджеру;
  - добавлена адресная строка в manager checklist: `Проверьте N — вычислено ботом, в прайсе нет`;
  - числа из фактов и из реплики клиента не помечаются.
- `scripts/run_telegram_dynamic_client_sim.py`: компактная metadata authoritative gate сохраняет `detail` и `span`.
- `tests/test_subscription_llm_draft_provider.py`, `tests/test_telegram_dynamic_client_sim.py`: NEG на дериватив, allow на факт/клиентское число, metadata.

Проверки:
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_subscription_llm_draft_provider.py -k "direct_path_derived_product_number or unpk_installment_bank_clarification or direct_path_unsupported_product_number"`
  - `4 passed, 428 deselected`
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_telegram_dynamic_client_sim.py -k authoritative_gate_compact_metadata_keeps_detail_and_span`
  - `1 passed, 519 deselected`
- Полный pytest после части 2:
  - `1 failed, 2897 passed, 2 skipped`
  - Тот же единственный FAIL: `tests/test_deal_aware_stage2_attribution.py::test_confidence_thresholds_post_recalibration`.

## Часть 3

Изменения:
- `src/mango_mvp/channels/subscription_llm.py`:
  - `TELEGRAM_ROUTE_RUBRIC` добавлен в `DIRECT_PATH_PILOT_PROFILE_DEFAULT_ON_FLAGS`;
  - `_route_rubric_enabled` теперь читает профиль через `_pilot_profile_flag_enabled`, явный `TELEGRAM_ROUTE_RUBRIC=0` поверх профиля выключает рубрику.
- `tests/test_subscription_llm_draft_provider.py`: тест профиля обновлен под новое решение Дмитрия.
- `product_data/telegram_dynamic_test_sets/pilot_smoke18_2026-06-10.jsonl`: добавлен smoke18 из `D1_audit_backlog/SCENARIOS_pilot_smoke15_2026-06-10.md` + addendum, сценарий 12 в новой формулировке.

Проверки:
- JSONL smoke18 валиден через `load_dynamic_sim_input`: 18 personas, simulator/judge specs присутствуют.
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_subscription_llm_draft_provider.py -k "route_rubric_enabled_by_pilot_gold_profile or route_rubric_prompt_off_golden"`
  - `2 passed, 430 deselected`
- Полный pytest после части 3:
  - `1 failed, 2897 passed, 2 skipped`
  - Тот же единственный FAIL: `tests/test_deal_aware_stage2_attribution.py::test_confidence_thresholds_post_recalibration`.

## Тестовая стабилизация

После трех частей полный pytest стабильно падал на одном AMO attribution тесте, потому что fixture с `deal_updated_at=2026-05-10T10:00:00+00:00` стал старше 30 дней относительно текущей даты `2026-06-10`, а код честно считает свежесть через `datetime.now(timezone.utc)`.

Изменение:
- `tests/test_deal_aware_stage2_attribution.py`: тест `test_confidence_thresholds_post_recalibration` замораживает `datetime` внутри `mango_mvp.deal_aware.deal_attribution` на `2026-05-25`, чтобы проверять именно пороги, а не текущую дату запуска.

Проверки:
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_deal_aware_stage2_attribution.py::test_confidence_thresholds_post_recalibration`
  - `1 passed`
- Финальный полный pytest:
  - `2898 passed, 2 skipped, 1 warning`

## Smoke18

Первый запуск:
- Out-dir: `runs/20260610_tz11_smoke18_rubric_on`
- Результат: невалидный инфраструктурный прогон (`turns=0`, `bot_direct_draft=0`).
- Причина: основной `~/.codex/config.toml` содержит `service_tier = "default"`, текущий `codex exec` принимает только `fast|flex`.
- Основной `~/.codex/config.toml` не менялся.

Валидный запуск:
- Временный `CODEX_HOME`: `/private/tmp/mango_codex_home_rubric_20260610_fast`.
- Out-dir: `runs/20260610_tz11_smoke18_rubric_on_fast`.
- Команда: `scripts/run_telegram_dynamic_client_sim.py`, `--parallel 4`, `--judge-prompt-version v9`, `TELEGRAM_DIRECT_PATH_PILOT_CONFIG=pilot_gold_v1`, snapshot r3.

Итог:
- `ok=true`
- `18 dialogs`, `48 turns`
- `6 PASS`, `9 PASS_WITH_NOTES`, `3 FAIL`
- `config_validity.invalid=false`
- `llm_calls.bot_direct_draft=46`
- `llm_calls.bot_semantic_output_verifier=61`
- `llm_calls.bot_faithfulness=0`
- `direct_path_rubric.rubric_enabled=48`
- `direct_path_rubric.rubric_regenerated=0`
- `direct_path_rubric.deferral_text_in_self=0`

## Semantic review

Статус:
- `formal_pass`: да. Кодовые NEG и smoke18 механически отработали; профиль включает рубрику; direct path, verifier и gate реально вызывались; финальный полный pytest зеленый.
- `semantic_pass`: не выставляю. В smoke18 есть `3 FAIL` и `9 PASS_WITH_NOTES`; по ТЗ разбор результатов делает архитектор по сырью.

Проверено вручную:
- Рубрика v1.1 не запрещает соседнее перечисление фактических цен, но запрещает новый расчет процентов/итогов.
- `derived_product_number` не заменяет текст менеджеру, а добавляет адресный checklist.
- Smoke18 включает контроль дериватива, PII, P0, рендера и KB-переходов.

Остаточные риски:
- Содержательные FAIL smoke18 не классифицированы здесь намеренно, чтобы не подменять регрейд архитектора.
