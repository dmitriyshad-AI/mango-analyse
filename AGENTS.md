# Mango Analyse: Project Instructions for Codex

## Language and Style

Всегда отвечай Дмитрию на русском языке, кратко и по делу, без сложных английских терминов.

## Subagents

Для сложных задач аудита, архитектуры, планирования, рефакторинга, проверки больших изменений и реализации крупных ТЗ можно запускать до 6 субагентов параллельно с максимальным разумным уровнем рассуждения `xhigh`.

Перед запуском субагентов кратко объясняй, какие части задачи им поручаешь. Не запускай субагентов для простых вопросов, коротких команд, мелких правок и задач, где параллельность не даёт пользы.

Если упёрся в лимит активных субагентов, сначала забери полезный результат старых, закрой неактуальных и только потом запускай новых.

## Hard Safety Boundaries

Нельзя без отдельного явного подтверждения Дмитрия:

- удалять файлы или папки;
- делать `git reset`, `git checkout`, `git clean` или другие разрушительные git-команды;
- менять `stable_runtime` DB/audio/transcripts;
- запускать ASR;
- запускать Resolve+Analyze по реальным данным;
- писать в AMO/CRM;
- писать в Tallanto;
- отправлять сообщения клиентам;
- запускать тяжелые batch/start/run-ui скрипты;
- запускать live-write скрипты;
- массово двигать, архивировать или чистить runtime-артефакты.

Если нужен live-доступ, запись во внешнюю систему или удаление, сначала остановись и попроси отдельное подтверждение.

## Source of Truth

Не восстанавливай актуальное состояние проекта из чата. Сначала читай:

1. `AGENTS.md`
2. `docs/PROJECT_NOW.md` (локальный generated-снимок текущей очереди, ветки, блокеров и свежих audit packs; если отсутствует или старше 24 часов — сначала запусти `python3 scripts/project_now.py`)
3. `docs/DECISIONS_LOG.md`
4. `docs/RUNBOOK.md`
5. актуальное ТЗ текущего блока
6. последние audit packs в `audits/_inbox/`
7. `docs/CURRENT_STATE.md` и `docs/ROADMAP.md` только как исторический контекст, если они явно свежее текущего ТЗ или нужны для проверки старого решения
8. `stable_runtime/CURRENT_RUNTIME.json` только для чтения, если это нужно для проверки runtime-указателей

Чат можно использовать только как дополнительный контекст, но не как источник правды.

## Task Queue

Очередь ТЗ живёт в `tasks/`:

- `tasks/_inbox_codex/` — новые ТЗ, не брать в работу без явного запроса или текущего приоритета.
- `tasks/_running/` — ровно те ТЗ, которые сейчас исполняются.
- `tasks/_done/` — завершённые ТЗ и отчёты.
- `tasks/_failed/` — остановленные ТЗ с причиной.

Перемещай ТЗ только через `python3 scripts/task_move.py`: `--take`, `--done`, `--fail`. Старый inbox не триажить массово без отдельной команды; для залежавшихся задач делай только отчёт `python3 scripts/task_stale_report.py`.

## Preflight

Перед крупной реализацией порядок такой:

1. `python3 scripts/project_now.py`
2. `python3 scripts/task_move.py --take <TZ.md>`
3. `python3 scripts/preflight.py --tz tasks/_running/<TZ.md>`

`preflight.py` должен остановить работу, если ТЗ не в `_running`, заявленные зоны пересекают запретные live/runtime пути, есть новая грязь вне зон ТЗ, `PROJECT_NOW.md` устарел, активный worktree не внесён в `docs/worktrees_registry.md`, или тест-команда из шапки не собирается в безопасном `--collect-only` режиме.

## Interfaces

Репо-локальные операционные инструменты:

- `scripts/project_now.py` — generated-снимок текущего состояния в `docs/PROJECT_NOW.md` (игнорируется git).
- `scripts/task_move.py` — единственный штатный перенос ТЗ между inbox/running/done/failed.
- `scripts/task_stale_report.py` — read-only отчёт о залежавшихся ТЗ.
- `scripts/make_audit_pack.py` — audit pack с ПДн-фильтром телефонов и email; `manifest.json` пишется последним.
- `scripts/preflight.py` — стоп-гейт перед крупной задачей.

## TZ Header

Для новых ТЗ добавляй машиночитаемую шапку в начале файла:

```text
Ветка: main
Зоны: scripts/, tests/, docs/, tasks/, AGENTS.md, .gitignore
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/...
Семантический-аудит: да/нет
```

Зоны должны быть минимальными. Запретные runtime/live-write зоны (`stable_runtime/`, `~/.codex`, AMO/Tallanto/CRM write, M1 queue, `runs/`, `transcripts/`) не включать без отдельного явного подтверждения Дмитрия.

## Graphify / карта репозитория

Граф/карта — чтобы быстро найти нужное место. Любой факт, число, имя правила, цитату промпта подтверждай чтением исходного файла, прежде чем строить на этом вывод или код. Карта не источник истины.

Если карта собрана не на текущей версии репозитория, вывод "такого узла/связи нет" по карте делать нельзя. Отсутствие проверяй поиском по сырью на текущей версии.

Утверждения карты о маршрутизации P0, разделении брендов, client-safe/manager-only фактах и защитных слоях всегда перепроверяй в исходниках; для P0/брендов/гвардов первым источником считать `src/mango_mvp/channels/rules_engine.py`.

`graphify-out/` — чувствительный локальный артефакт. Не коммитить, не передавать наружу, не использовать как runtime-истину.

На время пилота разрешён только структурный слой без облачной модели. Смысловой облачный слой включается только по отдельному явному подтверждению Дмитрия, после белого списка, ПДн-детектора и отчёта payload.

Graphify-сервер для диалогов работает только read-only через stdio. HTTP-режим, write-инструменты и исполнение команд через карту запрещены.

Graphify-навык или wrapper обязан возвращать путь к исходнику и баннер ревизии карты; при расхождении карты с `HEAD` каждый ответ должен явно писать: "карта построена на X, сейчас Y, проверяй в сырье".

## Current Working Rule

Основной цикл работы:

`аудит -> ТЗ -> реализация -> тесты -> audit pack -> коммит`

Для базы знаний, Telegram/email-черновиков, CRM/AMO/Tallanto-текстов и любых клиентских ответов этот цикл расширяется:

`аудит -> ТЗ -> реализация -> тесты -> semantic review -> audit pack -> коммит`

`quality_passed=true` и зеленые тесты означают только `formal_pass`. Нельзя писать "готово к использованию", пока не пройден смысловой аудит по `docs/SEMANTIC_REVIEW_RULES.md`.

Не начинать реализацию крупного блока без актуального ТЗ и понятных границ.

Не вести несколько крупных блоков параллельно в одном рабочем дереве. Исключение: независимые read-only аудиты без изменений.

## Frozen Legacy Bot Areas

Legacy-слои `subscription_llm_parts/policy_routing.py`, `channels/rules_engine.py` и `channels/answer_quality_rewriter.py` считаются замороженной legacy-лазанью для текущего Telegram-пилота. Не расширяй их без отдельного ТЗ: живой пилотный путь идёт через direct path в `src/mango_mvp/channels/subscription_llm_parts/provider.py:913` -> `_build_direct_path_draft()` при `_direct_path_enabled(context)`, а legacy-ветки используются только как совместимость/страховочная история.

## Git Discipline

Перед изменениями проверяй `git status --short`.

Не смешивай в один коммит:

- код;
- runtime-артефакты;
- документы уборки;
- live-write отчеты;
- внешние Excel/CSV;
- unrelated изменения пользователя.

Если рабочая папка грязная, работай только с файлами своего блока и не откатывай чужие изменения.

## Safe Tests

Базовая безопасная проверка:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest --collect-only -q
```

Для точечных тестов использовать:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q <tests>
```

Не запускать тесты или скрипты, которые требуют live AMO/Tallanto/CRM, ASR, R+A или реальный write в `stable_runtime`, если Дмитрий не подтвердил это отдельно.

## Audit Pack Rule

После значимого блока создавать audit pack в `audits/_inbox/<block>_<timestamp>/`:

- `implementation_notes.md`
- `changed_files.txt`
- `test_output.txt`
- `semantic_review.md` для базы знаний, ботов, CRM-текстов и клиентских ответов
- `risk_review.md`
- `backward_compatibility.md`

Для AMO/writeback блоков отдельно добавлять:

- dry-run отчет;
- snapshot/rollback contract;
- readback plan;
- список того, что не было записано live.
