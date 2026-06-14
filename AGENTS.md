# Mango Analyse: Project Instructions for Codex

## Language and Style

Всегда отвечай Дмитрию на русском языке, кратко и по делу, без сложных английских терминов.

Для сложных задач можно использовать до 6 субагентов с уровнем рассуждения `xhigh`. Для простых вопросов, коротких команд и мелких правок субагентов не запускать.

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
2. `docs/CURRENT_STATE.md`
3. `docs/DECISIONS_LOG.md`
4. `docs/ROADMAP.md`
5. `docs/RUNBOOK.md`
6. актуальное ТЗ текущего блока
7. последние audit packs в `audits/_inbox/`
8. `stable_runtime/CURRENT_RUNTIME.json` только для чтения, если это нужно для проверки runtime-указателей

Чат можно использовать только как дополнительный контекст, но не как источник правды.

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

## Current Main Priority

Текущий утвержденный порядок по ТЗ:

1. `G` - git-границы и рабочее состояние.
2. `A` - AMO pre-write snapshot и rollback.
3. `PBF` - красный post-backfill тест.
4. `B` - коммерческие поля в deal-aware AMO payload.
5. `C` - структурные возражения.
6. `D` - связь каталога вопросов и deal-aware quality gate.
7. `E` - customer timeline как read-only источник истории клиента.

Основное ТЗ:

`docs/TOP3_PRIORITY_FIXES_TZ_2026-05-15.md`
