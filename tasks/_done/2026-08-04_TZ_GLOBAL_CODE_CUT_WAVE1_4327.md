> DONE 2026-08-04 22:21 | ветка main | codex

> TAKE 2026-08-04 22:06 | ветка main | codex

Ветка: main
Зоны: src/mango_mvp/channels/customer_context_for_draft.py, src/mango_mvp/channels/web_chat_adapter.py, src/mango_mvp/channels/__init__.py, src/mango_mvp/existing_clients/amo_step2_scan.py, src/mango_mvp/existing_clients/amo_step3_contact_cards.py, src/mango_mvp/existing_clients/run_roots.py, src/mango_mvp/knowledge_base/drive_inventory.py, src/mango_mvp/insights/phase2_detectors.py, tests/test_customer_context_for_draft.py, tests/test_channels_web_chat_adapter.py, tests/test_channels_signals.py, tests/test_channels_feedback.py, tests/test_exact_runtime_dedup_contract.py, tests/test_existing_clients_amo_step2_scan.py, tests/test_existing_clients_amo_step3_contact_cards.py, tests/test_existing_clients_tz14_run_roots.py, tests/test_drive_inventory.py, tests/test_phase2_detectors.py, tasks/, audits/_inbox/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_draft_loop.py tests/test_run_amo_wappi_draft_loop.py tests/test_wappi_draft_loop_ops.py tests/test_bot_safe_runtime_context.py tests/test_customer_timeline_nightly_service.py tests/test_kc_knowledge_snapshot.py tests/test_build_kc_final_release.py tests/test_channels_signals.py tests/test_channels_feedback.py tests/test_exact_runtime_dedup_contract.py tests/test_adr003_regex_understanding_moratorium.py tests/test_graphify_structural.py tests/test_single_owner_registry.py
Семантический-аудит: да

# Глобальная резка кода, волна 1: удалить 4 327 строк старых контуров

## Образ результата и бизнес-польза

В репозитории остаются только владельцы функций, которые нужны целевому ИИ-сотруднику:
`Wappi -> черновик AMO -> менеджер`, актуальная база знаний и Customer Timeline.
Разработчик больше не выбирает между старым и новым сборщиком контекста, старым
AMO-конвейером, старым инвентарём Drive или неиспользуемым web-chat. Удаление не
меняет ни одного текста черновика, факта памяти, события Timeline или текущего
KB-релиза.

## Зафиксированная исходная точка

- HEAD: `7fd6761b`;
- Graphify: свежая структурная карта этого репозитория на `f1ea0f88`; после
  коммита attendance кодовые связи кандидатов не менялись;
- пять независимых read-only аудитов нашли первую волну, два аудитора уже
  подтвердили кандидатов по сырью;
- точный объём прямых файлов: `4 327` строк, из них `3 099` строк `src` и
  `1 228` строк прямых тестов.

## Удалить целиком

1. `channels/customer_context_for_draft.py` и прямой тест: старый test-only
   сборщик; живой путь использует `customer_timeline/bot_safe_runtime_context.py`.
2. `existing_clients/amo_step2_scan.py`, `amo_step3_contact_cards.py`,
   `run_roots.py` и прямые тесты: их CLI удалены как obsolete в `d75c47e2`;
   `amo_step1_snapshot.py` оставить. Старые callback-task и family-note CSV не
   переносить: действующие владельцы разделены между nightly Timeline,
   `derived_signals`/`objections` и manager draft; старые write/CSV-пути сняты.
3. `knowledge_base/drive_inventory.py` и прямой тест: старый test-only
   инвентарь; текущий KB-релиз использует `fact_registry.py` и текущие builders.
4. `insights/phase2_detectors.py` и прямой тест: старый test-only анализатор;
   runtime-ветка снята решением D-092. Текущие owners покрывают conversation
   intent и Timeline objections; старую regex-taxonomy тревог сознательно не
   сохранять и не объявлять эквивалентной заменой.
5. `channels/web_chat_adapter.py` и прямой тест: D-095/D-105 исключают web-chat
   из целевой архитектуры; удалить re-export из `channels/__init__.py` и один
   web-chat-only тест из `test_channels_signals.py`. Сквозной feedback-тест не
   удалять: заменить только web-chat вход на прямой `ChannelMessage`, сохранив
   проверки `draft -> decision -> feedback`, `live_send=false`, `write_crm=false`.
6. Удалить test-only контракт `customer_context_for_draft.int_or_zero` из
   `test_exact_runtime_dedup_contract.py`; канонический `pilot_context.int_or_zero`
   остаётся и продолжает проверяться своими живыми вызывающими.

## Перед удалением

1. Перестроить Graphify на текущем HEAD.
2. Для каждого файла подтвердить ноль callers вне его прямых тестов, старых
   документов и package re-export.
3. Проверить `pyproject.toml`, `scripts/`, launchd/config и пять source-of-truth
   документов на отсутствие entry point.
4. Зафиксировать baseline полной сборки тестов через `--collect-only` и целевой
   тест-команды.

## Не делать

- не трогать P0, output floor, модельное понимание, память, Timeline и KB-факты;
- не удалять `existing_clients/amo_step1_snapshot.py`;
- не объединять разные `env_bool`, `upsert_session`, роли или write helpers;
- не включать в эту волну спорный `crm_writeback_population_recall.py`;
- не удалять старые документы: они не runtime и будут отдельной документальной волной;
- не добавлять новый compatibility wrapper, feature flag или replacement module.

## Ломающие проверки

1. Полный `pytest --collect-only` не содержит ImportError.
2. `rg` не находит удалённые модули в `src/`, `scripts/`, `tests/`,
   `pyproject.toml`, кроме явного теста отсутствия, если он действительно нужен.
3. Импорт `mango_mvp.channels`, draft-loop, Timeline memory, fact registry и
   текущего KB builder проходит.
4. Точечные Wappi/AMO/Timeline/KB тесты из шапки остаются зелёными.
5. Отрицательный import-bomb до удаления подтверждает, что живые сквозные тесты
   не импортируют кандидатов; после удаления тот же набор зелёный.
6. Ломатель отдельно ищет динамические импорты и строковые CLI-вызовы.

## Приёмка

- удалено не менее `4 327` строк прямых файлов и больше строк, чем добавлено;
- новых рабочих файлов, функций, флагов, зависимостей и LLM-вызовов — 0;
- formal_pass: полный collect и целевой набор зелёные;
- semantic_pass/business_pass: фактический Wappi -> AMO draft, Timeline memory и
  текущий KB-релиз не меняются;
- breaker_pass: нет скрытого caller, entry point или единственной бизнес-функции;
- один audit pack с точным списком удалений, тестами и остаточными рисками;
- один отдельный коммит, затем push в `origin/main` и `yandex/main`.

## СТОП

Остановить удаление конкретного кандидата, если найден production caller,
динамический импорт, единственная актуальная бизнес-функция или тест живого пути
меняет поведение. Остальные доказанно независимые кандидаты удалить той же волной.
