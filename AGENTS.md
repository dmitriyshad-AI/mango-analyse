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

## Current Working Rule

Основной цикл работы:

`аудит -> ТЗ -> реализация -> тесты -> audit pack -> коммит`

Не начинать реализацию крупного блока без актуального ТЗ и понятных границ.

Не вести несколько крупных блоков параллельно в одном рабочем дереве. Исключение: независимые read-only аудиты без изменений.

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

