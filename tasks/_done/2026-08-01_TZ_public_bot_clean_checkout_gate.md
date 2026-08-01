> DONE 2026-08-01 11:15 | ветка codex/critical-gates-integration-20260731 | codex

> TAKE 2026-08-01 11:08 | ветка codex/critical-gates-integration-20260731 | codex

Ветка: codex/critical-gates-integration-20260731
Зоны: scripts/run_telegram_public_pilot_bots.py, tests/test_telegram_public_pilot_bots.py, docs/RUNBOOK.md, docs/worktrees_registry.md, tasks/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_telegram_public_pilot_bots.py
Семантический-аудит: да

# Запрет запуска публичного Telegram из грязного дерева

## Цель

Публичный Telegram-бот запускается только из чистого Git-worktree на ревизии,
содержащей денежный защитный пол `ca1c9ce5`.

## Минимальное изменение

- Переиспользовать существующий `assert_public_bot_minimum_safe_revision`.
- После проверки ревизии выполнить fail-closed `git status --porcelain`.
- Блокировать staged, unstaged и untracked изменения без вывода имён файлов.
- Не менять маршрут, генерацию ответа, вызовы модели или live-конфигурацию.

## Готово, когда

- чистый текущий worktree проходит;
- грязный tracked-файл блокирует запуск;
- untracked-файл блокирует запуск;
- `mango_mvp` из другого worktree блокирует запуск;
- ошибка команды Git блокирует запуск;
- старый SHA по-прежнему блокируется;
- live не запускался.

## Приёмка

Целевой тест зелёный; отдельный read-only зонд на грязном основном worktree
получает отказ до создания Telegram-приложения.

## СТОП

- Любой live-запуск или сетевое сообщение.
- Необходимость менять генерацию ответа или защитные полы.
