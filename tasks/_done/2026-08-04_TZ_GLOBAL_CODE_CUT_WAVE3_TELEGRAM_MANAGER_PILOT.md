> DONE 2026-08-04 23:23 | ветка main | codex

> TAKE 2026-08-04 22:57 | ветка main | codex

Ветка: main
Зоны: pyproject.toml, requirements.txt, uv.lock, src/mango_mvp/channels/__init__.py, src/mango_mvp/channels/telegram_adapter.py, src/mango_mvp/channels/telegram_bot_polling.py, src/mango_mvp/channels/telegram_business_runtime.py, src/mango_mvp/channels/telegram_manager_inbox.py, src/mango_mvp/channels/telegram_native_draft.py, src/mango_mvp/channels/telegram_pilot_store.py, src/mango_mvp/channels/telegram_pilot_metrics.py, src/mango_mvp/channels/telegram_pilot_reporting.py, src/mango_mvp/channels/telegram_pilot_p0_register.py, scripts/telegram_manager_draft_pilot.py, scripts/run_telegram_pilot_concurrency_smoke.py, scripts/build_telegram_pilot_eval_pack.py, scripts/telegram_pilot_daily_report.py, scripts/build_telegram_pilot_daily_report.py, scripts/import_telegram_pilot_feedback.py, tests/test_channels_telegram_adapter.py, tests/test_telegram_bot_polling.py, tests/test_channels_telegram_business_runtime.py, tests/test_telegram_manager_inbox.py, tests/test_channels_telegram_native_draft.py, tests/test_telegram_manager_draft_pilot_script.py, tests/test_telegram_pilot_store.py, tests/test_telegram_pilot_metrics.py, tests/test_telegram_pilot_journal_report.py, tests/test_telegram_pilot_feedback_import.py, tests/test_telegram_pilot_p0_register.py, tests/test_telegram_pilot_concurrency_smoke.py, tests/test_telegram_pilot_eval_pack.py, tests/test_exact_runtime_dedup_contract.py, tests/test_single_owner_registry.py, tests/test_adr003_regex_understanding_moratorium.py, tests/fixtures/adr003_runtime_channel_regex_snapshot.json, tests/test_kb_r4_1_owner_gap_answers.py, tasks/, audits/_inbox/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_draft_loop.py tests/test_run_amo_wappi_draft_loop.py tests/test_wappi_draft_loop_ops.py tests/test_business_journey_traces_wave1.py tests/test_bot_safe_runtime_context.py tests/test_telegram_pilot_context_builder.py tests/test_channels_telegram_history.py tests/test_wappi_history_import_to_timeline.py tests/test_subscription_llm_draft_provider.py tests/test_output_verification_floor_contract.py tests/test_output_verification_floor_regressions.py tests/test_exact_runtime_dedup_contract.py tests/test_single_owner_registry.py tests/test_adr003_regex_understanding_moratorium.py tests/test_kb_r4_1_owner_gap_answers.py
Семантический-аудит: да

# Глобальная резка кода, волна 3: удалить Telegram manager/pilot sidecar

## Образ результата и бизнес-польза

У менеджера остаётся один рабочий интерфейс: черновик в AMO, сформированный из
входящего Wappi. В проекте больше нет параллельного Telegram-бота менеджера,
TDLib-черновиков, отдельной SQLite памяти пилота, второго P0-регистра и второго
набора отчётов качества.

Это сокращает выбор ложных архитектурных путей и освобождает разработку для
улучшения полезности AMO-черновика. Текущий контекст модели, импорт Telegram/Wappi
истории в Customer Timeline, provider и защитные полы сохраняются.

## Зафиксированный кандидат

- девять старых `src` модулей;
- шесть scripts manager/pilot/eval/report;
- тринадцать прямых тестов;
- прямой объём: 9 017 строк до сопутствующей очистки;
- package re-export, зависимость `python-telegram-bot`, один test-only dedup
  контракт, regex budget/snapshot и один KB runner-list должны быть очищены.

## Канонические владельцы, которые обязаны остаться

1. Входящие, дедупликация, identity, черновик и AMO note:
   `integrations/draft_loop.py` + `run_amo_wappi_draft_loop.py`.
2. Правки менеджера и таблица качества:
   manager edit log + `wappi_draft_loop_ops.py quality-table`.
3. P0, бренд, факты, числа и анти-выдумка:
   provider/direct path/output verification floor.
4. Контекст KB для модели: `telegram_pilot_context_builder.py` — сохранить.
5. История Telegram и связи клиентов в Timeline: `telegram_history.py` и
   Wappi history import — сохранить.

## Перед удалением

1. Graphify и raw source должны показать ноль production callers вне замкнутого
   manager/pilot контура, его re-export и прямых тестов.
2. Проверить pyproject/requirements/uv.lock, launchd/cron/processes и dynamic imports.
3. Архитектор, ломатель, бизнес-аудитор и Claude/Fable независимо проверяют ТЗ.
4. Снять baseline collect, прямых тестов кандидата и целевого Wappi/Timeline/P0 набора.

## Удалить и упростить

1. Удалить девять старых модулей, шесть scripts и тринадцать прямых тестов.
2. Удалить их package imports и `__all__` из `channels/__init__.py`.
3. Удалить `python-telegram-bot` из pyproject/requirements и штатно обновить uv.lock,
   только если raw source подтверждает отсутствие оставшихся импортов зависимости.
4. Удалить семь regex удалённого reporting-модуля из бюджета и frozen snapshot;
   не повышать другие бюджеты.
5. Удалить test-only store helper из exact-runtime test и снизить single-owner
   baseline до фактического значения.
6. Убрать concurrency smoke из списка текущих KB runners; Wappi runner и M1
   dynamic/P0 измерители сохранить.

## Не делать

- не удалять `telegram_pilot_context_builder.py`, `telegram_history.py` или их тесты;
- не трогать Wappi draft-loop, provider, Timeline, P0/output floors и KB-факты;
- не удалять исторические runtime SQLite/логи;
- не писать replacement wrapper, feature flag или новую телеметрию;
- не запускать live и не писать в AMO/Tallanto/CRM.

## Ломающие и бизнес-проверки

1. Import-bomb удаляемых модулей не ломает Wappi, provider, Timeline history и P0 tests.
2. Wappi всё ещё создаёт только AMO draft, `sends_client_replies=False`.
3. Manager edit classification и quality-table остаются зелёными без pilot store.
4. Память своего клиента доходит, чужая не доходит; P0 блокируется.
5. Telegram/Wappi history продолжает импортироваться и связываться с Timeline.
6. `import mango_mvp.channels` проходит без удалённых re-export.
7. Regex moratorium показывает уменьшение на семь, а не перенос regex в другой файл.
8. Full collect и полный pytest не получают новых падений относительно HEAD 8311ba89.

## Приёмка

- удалено не менее 9 000 прямых строк и больше строк, чем добавлено;
- `python-telegram-bot` больше не является зависимостью проекта;
- новых рабочих файлов, функций, флагов, зависимостей и LLM-вызовов: 0;
- formal, semantic, business и breaker checks пройдены;
- один audit pack, один коммит, push в origin/main и yandex/main.

## СТОП

Остановить конкретный срез, если найден production caller или единственная
бизнес-функция, которой нет в Wappi/AMO. Не переносить функцию автоматически:
сначала доказать, нужна ли она целевому ИИ-сотруднику.
