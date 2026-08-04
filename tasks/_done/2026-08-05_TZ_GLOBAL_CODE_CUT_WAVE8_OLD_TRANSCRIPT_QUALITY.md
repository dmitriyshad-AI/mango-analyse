> DONE 2026-08-05 02:39 | ветка main | codex

> TAKE 2026-08-05 02:24 | ветка main | codex

Ветка: main
Зоны: src/mango_mvp/quality/, scripts/, tests/, docs/SCRIPT_SAFETY_MATRIX.md, tasks/, audits/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_non_conversation_quality.py tests/test_analyze.py tests/test_analysis_schema.py tests/test_crm_writeback_quality_detector.py tests/test_transcript_quality_baseline.py tests/test_transcript_quality_stage14_comparison.py tests/test_transcript_quality_stage15_export_gate.py tests/test_productization_call_processing_readiness.py tests/test_productization_current_runtime.py
Семантический-аудит: нет

# Волна 8: удалить завершённые контуры проверки старых расшифровок

## Проблема

В мае 2026 года проект последовательно создал большой контур разовой проверки, LLM-рецензирования, согласования и backup/apply старых расшифровок. Цикл завершён, но исполняемый аппарат остался рядом с текущим конвейером и выглядит как альтернативный живой механизм.

Текущий runtime не вызывает удаляемые файлы. Живой анализ использует `AnalyzeService` и `quality/non_conversation.py`; CRM writeback использует отдельный `crm_writeback_quality_detector.py`. Повторная проверка выявила действующий контракт: `CURRENT_RUNTIME.json` и `call_processing_readiness.py` требуют Stage 15. Поэтому `transcript_quality_baseline.py`, Stage 14/15, их три scripts и три теста сохраняются.

## Образ результата и бизнес-польза

В проекте остаётся один понятный текущий путь определения несодержательного звонка, один текущий CRM quality gate и воспроизводимый Stage 15, на который ссылается runtime. Удаляются только завершённые review/backfill-планы. Менеджерские данные и Customer Timeline сохраняют прежнее поведение; сопровождать и ошибочно запускать более 10 тысяч строк исторического контура больше не нужно.

## Проверка существующей реализации до изменений

1. Graphify на `b9bb4f1d` указывает текущий путь на `src/mango_mvp/services/analyze.py` и `src/mango_mvp/quality/non_conversation.py`.
2. Поиск точных импортов удаляемых модулей на `b9bb4f1d` находит только их собственные scripts/tests и внутренние связи семейства.
3. В `docs/RUNBOOK.md`, `docs/PROJECT_NOW.md`, других `tasks/_running`, `deploy`, `pyproject.toml`, `cli.py`, `__main__.py` и установленных LaunchAgents ссылок нет.
4. До удаления собирается `5207` тестов.
5. `stable_runtime/CURRENT_RUNTIME.json` указывает на Stage 15, поэтому первоначальное предложение удалить Stage 14/15 отклонено как небезопасное.

## Рассмотренные варианты

1. **Выбран:** удалить замкнутый завершённый review/backfill-контур и его собственные тесты, сохранив текущих владельцев и Stage 15.
2. Оставить как есть: нулевая бизнес-польза и постоянный риск принять исторический pipeline за runtime.
3. Перенести код в архив: сохраняет тот же объём и ложный интерфейс; Git уже хранит историю.

## Точный объём

Удалить:

- 15 модулей `src/mango_mvp/quality/`: все `hard_gate_*` и старые `transcript_quality_*`, кроме baseline;
- 18 одноразовых scripts с review/backfill/hard-gate и `finalize_manual_non_conversation_tail.py`;
- 11 собственных тестов удаляемого семейства;
- 3 устаревшие строки этих scripts из `docs/SCRIPT_SAFETY_MATRIX.md`.

Итого: 44 файла и 10 235 строк кода/тестов плюс 3 строки устаревшего каталога. `DELETE` данных и runtime-артефактов не выполнять.

Обязательно сохранить:

- `quality/non_conversation.py` и его экспорт из `quality/__init__.py`;
- внутренний guard в `services/analyze.py`;
- `crm_writeback_quality_detector.py`, `crm_writeback_frozen_corpus.py`, `crm_writeback_population_recall.py`;
- текущие calls/Customer Timeline потребители `is_non_contentful_call_record`.
- `transcript_quality_baseline.py`, `stage14_quality_comparison.py`, `stage15_export_quality_gate.py`, три их scripts и три теста;
- Stage 15 gates в `productization/call_processing_readiness.py` и текущий runtime-контракт.

## Приёмка и критерии готовности

1. Удалены ровно 44 заявленных файла и 10 235 строк кода/тестов; новых runtime-файлов, флагов и зависимостей нет.
2. Поиск точных имён не находит исполняемых ссылок вне исторических документов.
3. Количество собранных тестов уменьшается ровно на 23 теста удалённого семейства: `5207 -> 5184`.
4. Тест-команда из шапки зелёная.
5. Импорт `AnalyzeService`, `quality.non_conversation`, calls/Timeline и CRM writeback проходит.
6. Полный pytest не получает новых падений относительно известной базовой линии из 8 KB-зависимых падений.
7. Независимые архитектор и ломатель подтверждают отсутствие runtime- и бизнес-регрессии.
8. Audit pack содержит команды, сырые результаты, риски, совместимость и самодекларацию строк.

## СТОП

Остановиться без удаления, если найден хотя бы один вызов из текущего runtime/CLI/deploy, если живой `AnalyzeService` зависит от удаляемого модуля, если collect уменьшается не на 23 теста, если Stage 15 перестаёт воспроизводиться либо если полный pytest получает новое падение сверх зафиксированной базовой линии.

## Бритва

Изменение добавляет 0 строк нетестового кода и удаляет 10 235. Это минимум: живой механизм и обязательный Stage 15 не переписываются, второй преемник не создаётся.
