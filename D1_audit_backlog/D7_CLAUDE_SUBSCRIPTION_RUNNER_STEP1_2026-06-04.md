# D7 Step 1 — Claude CLI через подписку Дмитрия

Дата: 2026-06-04
Ветка: codex/d7-claude-first-class-brain
База: 176e6336
Коммит предыдущего дизайна: 9d85951e

## Цель

Починить запуск Claude CLI для режима подписки Дмитрия до A/B. Бизнес-гейты, логика бота, база знаний и конвейер качества не менялись.

## Что подтвердилось

`claude --help` прямо говорит, что `--bare` не читает обычную keychain/OAuth-сессию и работает только через `ANTHROPIC_API_KEY` или `apiKeyHelper`. Локальная проверка ранее дала `Not logged in` для `claude --bare -p ...`.

Значит гипотеза подтверждена в основной части: `--bare` несовместим с текущей залогиненной подпиской Дмитрия.

Уточнение: остановка Opus на `$0.05` сама по себе не доказывает API-биллинг, потому что это был явно заданный лимит `--max-budget-usd`. Надёжный вывод другой: для подписки нельзя использовать `--bare`; обычный `claude -p` без `--bare` работает.

## Что изменено

Файл: `scripts/run_telegram_dynamic_client_sim.py`

1. Добавлен `auth_mode` для Claude-команды:
   - `subscription` по умолчанию;
   - `bare` оставлен как явный режим для API/helper, если он когда-нибудь понадобится.

2. В subscription-режиме команда больше не содержит `--bare`.

3. Изоляция сохранена:
   - `-p`;
   - `--output-format text`;
   - `--tools ""`;
   - `--strict-mcp-config`;
   - `--mcp-config '{"mcpServers":{}}'`;
   - `--no-session-persistence`;
   - `--disable-slash-commands`;
   - `--permission-mode plan`;
   - `--effort <level>`.

4. Исправлен пустой MCP-конфиг. Старое значение `"{}"` текущий Claude CLI считает невалидным. Теперь используется валидный пустой конфиг `{"mcpServers":{}}`.

5. Добавлен CLI-флаг:

```bash
--claude-auth-mode subscription|bare
```

По умолчанию: `subscription`.

## Было

```bash
claude -p --bare --model <model> --output-format text --tools "" --strict-mcp-config --mcp-config "{}" --no-session-persistence --disable-slash-commands --permission-mode plan --effort <level>
```

## Стало

```bash
claude -p --model <model> --output-format text --tools "" --strict-mcp-config --mcp-config '{"mcpServers":{}}' --no-session-persistence --disable-slash-commands --permission-mode plan --effort <level>
```

## Живая проверка

Проверка выполнялась через тот же `ClaudeJsonModel`, который использует раннер.

Результат:

```text
{'model': 'claude-sonnet-4-6', 'has_bare': False, 'ok': True, 'probe': 'subscription_json'}
{'model': 'claude-opus-4-8', 'has_bare': False, 'ok': True, 'probe': 'subscription_json'}
```

Вывод:

- `claude-sonnet-4-6` работает через subscription-режим;
- `claude-opus-4-8` работает через subscription-режим;
- `--bare` отсутствует;
- JSON парсится штатно.

## Тесты

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_telegram_dynamic_client_sim.py
40 passed in 1.05s
```

Точечная проверка до полного файла:

```text
4 passed
```

## Остаточные риски

1. `bare`-режим оставлен, но не должен использоваться для подписки Дмитрия.
2. Маскирование stderr/секретов ещё не реализовано; это отдельный следующий инфраструктурный шаг перед массовыми A/B.
3. Полный pytest не запускался на этом шаге, потому что изменён только runner/test раннера.

## Вердикт

Шаг 1 закрыт: Claude CLI теперь может запускаться через залогиненную подписку Дмитрия без `--bare`, и обе целевые модели проходят tiny JSON-пробу.
