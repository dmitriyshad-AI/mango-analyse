# Claude CLI Review Setup

Дата: 2026-05-18.

Цель: дать Codex способ вызывать Claude CLI как независимого смыслового ревьюера без ручного копирования папок.

## Установленные файлы

- `CLAUDE.md` — текущая правда проекта для Claude CLI.
- `PROJECT_HISTORY.md` — история решений и итераций.
- `.claude/agents/kb-reviewer.md` — роль независимого ревьюера.
- `.claude/commands/kb-review.md` — read-only команда аудита по явному пути.

## Почему путь должен быть явным

В `product_data/knowledge_base/` рядом лежат:

- основной релиз;
- handoff;
- bot pack;
- employee pack;
- smoke input;
- smoke fake;
- smoke codex.

Поэтому Claude CLI не должен выбирать «последнюю папку» автоматически. Всегда передавать конкретный путь.

Главный путь для аудита бота:

```text
product_data/knowledge_base/kb_release_20260518_v3_2_bot_pack
```

## Проверенная команда запуска

Важно: после опций нужен разделитель `--`, иначе `--allowedTools` / `--disallowedTools` могут съесть текст промпта.

```bash
claude -p \
  --model opus \
  --effort xhigh \
  --max-budget-usd 1.50 \
  --allowedTools "Read,Grep,Glob,Bash" \
  --disallowedTools "Edit,Write" \
  -- \
  "/kb-review product_data/knowledge_base/kb_release_20260518_v3_2_bot_pack"
```

## Результаты теста

Быстрый тест контекста:

```bash
claude -p --model opus --effort xhigh --max-budget-usd 0.25 \
  "Прочитай CLAUDE.md и ответь одной фразой: какое главное правило проекта Mango Analyse сейчас?"
```

Claude CLI ответил корректно: главное правило — не смешивать УНПК МФТИ и Фотон.

Полный тест `/kb-review`:

```text
audits/_inbox/claude_cli_kb_review_20260518T195004Z/claude_review.md
```

Вердикт:

```text
PASS_WITH_NOTES
formal_pass: да
semantic_pass: да
pilot_ready: да с условиями
production_ready: нет
```

## Что нашёл Claude CLI

Не блокирует внутренний пилот на сотрудниках, но нужно закрыть до клиентского пилота:

1. Часть `client_safe_text` выглядит как машинный обрывок.
2. Несколько скидок не имеют условия прямо в клиентской строке.
3. 241 чувствительный к дате факт имеет только `freshness_check_date`, без `valid_until`.
4. Нужно проверить, что runtime бота реально читает `manifest.safety.client_send=false`.
5. Часть строк в `post_filter_registry.json` выглядит как описание паттерна, а не буквальная фраза.

## Правило дальнейшей работы

После каждой новой сборки базы знаний:

1. Codex запускает свои тесты и semantic gate.
2. Codex собирает audit pack.
3. Codex вызывает Claude CLI по явному пути через `/kb-review`.
4. Все подтверждённые смысловые замечания переводятся в тест, gate, чек-лист или отдельный ТЗ-блок.
