---
description: Read-only смысловой аудит базы знаний по явному пути
argument-hint: <artifact-path>
---

Проведи read-only аудит базы знаний или пакета по пути:

```text
$ARGUMENTS
```

Если путь не указан, остановись и попроси передать конкретный путь. Не выбирай последнюю папку автоматически.

## Контекст

Сначала прочитай:

1. `CLAUDE.md`
2. `.claude/agents/kb-reviewer.md`
3. только затем проверяемый путь `$ARGUMENTS`

## Запреты

Ничего не меняй в проверяемом пакете, коде, тестах, docs, `stable_runtime`, AMO, Tallanto или CRM.

Единственное разрешённое место записи:

```text
audits/_inbox/claude_cli_kb_review_<timestamp>/
```

## Что сделать

1. Определи тип пакета: bot pack, employee pack, handoff/release или другой audit artifact.
2. Проверь формальные признаки: manifest, quality_report, semantic_review, факты, правила брендов, контракт использования.
3. Если это handoff/release с `facts_registry.jsonl`, можно запустить безопасный gate:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/run_kb_semantic_review.py --release-dir "$ARGUMENTS"
```

4. Проведи смысловой аудит по правилам `.claude/agents/kb-reviewer.md`.
5. Сохрани `claude_review.md`.
6. Верни краткий итог:

```text
Verdict: PASS / PASS_WITH_NOTES / BLOCKED
formal_pass: да/нет/не проверялось
semantic_pass: да/нет
pilot_ready: да/нет/да с условиями
Главные замечания:
- ...
Отчёт: audits/_inbox/claude_cli_kb_review_<timestamp>/claude_review.md
```
