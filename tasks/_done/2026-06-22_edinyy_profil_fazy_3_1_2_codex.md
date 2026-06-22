# Единый профиль и самопроверка чекера: фазы 3, 1, 2

Дата: 2026-06-22  
Исполнитель: Codex  
Ветка: `codex/tz-profile-selfcheck`  
База: `8ffb752dfde5ddf535ea44cf19e008f09c803f65`

## Блокер по сырью

Проверено на `8ffb752` до реализации:

- `TELEGRAM_A_FREE_NUMBER_GATE` и `TELEGRAM_STEP4_NUMBER_GROUNDING` живут в `src/mango_mvp/channels/dialogue_contract_pipeline.py`; в `src/mango_mvp/channels/subscription_llm_parts/direct_path.py` и в direct-ветке `_build_direct_path_draft()` нет обращений к этим флагам.
- В `_build_direct_path_draft()` direct-flow идёт через `_apply_direct_path_model_p0_route`, `apply_assumed_scope_guard`, `apply_semantic_output_verifier`, `apply_authoritative_output_gate`; числовые гейты туда не входят.
- По `TELEGRAM_SEMANTIC_DIAGNOSIS_GUARD` уточнение: он не строго "только в `dialogue_contract_pipeline.py`" — есть non-direct использования в `provider.py`/`post_layers.py`, но в direct-ветке `_build_direct_path_draft()` его нет. Для живого direct-path добавление этого флага в профиль также является no-op.

Вывод: живой бот на direct-path не получает защиты от добавления этих трёх флагов в профиль. Фаза 5 не выполнялась.

## Что сделано

Фаза 3:

- `scripts/check_public_bot_live.py`: убран безусловный `setdefault("TELEGRAM_DIRECT_PATH_PILOT_CONFIG", "pilot_gold_v1")`.
- Чекер теперь читает heartbeat живого public bot, проверяет свежесть `last_cycle_at`, живость `pid`, статус `polling`, бренд, `effective_profile` и обязательные guards.
- Добавлен режим `--smoke-force-profile`: профиль прокидывается только в локальный temp-context smoke-прогона, без записи в глобальный `os.environ`.

Фаза 1:

- Добавлен общий helper `src/mango_mvp/channels/pilot_profile_runtime.py`.
- `ensure_canonical_pilot_profile()` работает только при `ENFORCE_CANONICAL_PROFILE=1`, default OFF.
- Семантика `setdefault`: пустой профиль заполняется `pilot_gold_v1`, явный непустой операторский override не затирается и даёт WARNING.
- Helper подключён в public bot, manager draft, AMO Wappi draft loop и M1 env materialization.
- Старый `setdefault` в `scripts/run_amo_wappi_draft_loop.py` заменён единым helper-ом.
- `write_local_env_file` дописывает `TELEGRAM_DIRECT_PATH_PILOT_CONFIG=pilot_gold_v1` в шаблон.

Фаза 2:

- Public bot startup порядок: env-файл -> sync в `os.environ` -> `ensure_canonical_pilot_profile()` -> самопроверка -> polling.
- Самопроверка fail-closed при включённом `ENFORCE_CANONICAL_PROFILE=1`, если профиль не ровно `pilot_gold_v1` или обязательные guards выключены.
- Булевы алиасы вроде `on`/`да` для канонического профиля считаются красным состоянием.
- Quality-флаги не валят старт, только попадают в warnings.
- `write_public_bot_heartbeat` расширен без второго файла.

## Heartbeat

Схема heartbeat public bot: `public_pilot_bot_heartbeat_v2_2026_06_21`.

Добавленные поля:

- `effective_profile`
- `draft_path`
- `active_guards`

Старые поля сохранены: `schema_version`, `status`, `last_cycle_at`, `pid`, `brands`, `event`, `summary`.

Подробности по каждому черновику (`draft_path`, `applied_guards`) пишутся в store/log payload, не в heartbeat.

## Изменённые файлы

- `src/mango_mvp/channels/pilot_profile_runtime.py`
- `scripts/check_public_bot_live.py`
- `scripts/run_telegram_public_pilot_bots.py`
- `scripts/telegram_manager_draft_pilot.py`
- `scripts/run_amo_wappi_draft_loop.py`
- `scripts/m1_watcher.py`
- `tests/test_check_public_bot_live.py`
- `tests/test_telegram_public_pilot_bots.py`
- `tests/test_run_amo_wappi_draft_loop.py`
- `tests/test_m1_watcher.py`

## Тесты

Команды:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m py_compile src/mango_mvp/channels/pilot_profile_runtime.py scripts/run_telegram_public_pilot_bots.py scripts/check_public_bot_live.py scripts/telegram_manager_draft_pilot.py scripts/run_amo_wappi_draft_loop.py scripts/m1_watcher.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_check_public_bot_live.py tests/test_telegram_public_pilot_bots.py tests/test_run_amo_wappi_draft_loop.py tests/test_m1_watcher.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q
git diff --check
```

Результат:

- py_compile: PASS
- focused pytest: `96 passed`
- full pytest: `3494 passed, 5 skipped, 1 warning in 74.33s`
- diff check: PASS

Единственный warning: `urllib3` про LibreSSL, не связан с изменениями.

## Read-only и границы

- Live Telegram bot не перезапускался и не останавливался.
- AMO/Tallanto/stable_runtime/M1 не трогались.
- Live write/PATCH/POST не выполнялись.
- Секреты и токены не выводились.
- Фазы 4 и 5 не выполнялись.
- `ENFORCE_CANONICAL_PROFILE` не включён по умолчанию.
- Состав `pilot_gold_v1` не менялся.
- `no_auto_send`/`manager_approval_required` не ослаблялись.

## ACK

ACK: фазы 3+1+2 реализованы на отдельной ветке, тесты зелёные, STOP на регрейд Claude #1.
