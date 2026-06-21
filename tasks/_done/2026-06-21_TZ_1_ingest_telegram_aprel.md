> DONE 2026-06-21 03:23 | ветка codex/tz1-telegram-aprel | codex

> TAKE 2026-06-21 03:00 | ветка codex/tz1-telegram-aprel | codex

Ветка: codex/tz1-telegram-aprel
Зоны: scripts/import_telegram_export_to_timeline.py, src/mango_mvp/customer_timeline/, tests/, tasks/, audits/_inbox/, docs/worktrees_registry.md, .gitignore
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest --collect-only -q
Семантический-аудит: да

# ТЗ (1) — Ингест апрельского Telegram-экспорта в память

Дата: 2026-06-21. Исходное ТЗ: `/Users/dmitrijfabarisov/Claude Projects/Foton/2026-06-21_TZ_1_ingest_telegram_aprel.md`.

## Цель

Влить уже имеющийся Telegram-экспорт в тестовую копию памяти клиента теми же правилами привязки и безопасности, что и письма.

## Вход

- Экспорт: `telegram_exports (2)/local_vm_2024-04-01_max`.
- Ожидаемо: 725 телефонов, `crm_contacts.csv`, 1653 диалога, 13223 сообщения.
- Бренд экспорта указан Дмитрием: `unpk`.

## Правила загрузки

- Только тестовая копия, не боевая база.
- Источник read-only.
- `source_system=telegram_history`.
- Лента manager-only: `allowed_for_bot=0`.
- Первичная привязка по телефону из `_max/dialogs.jsonl` и `_max/crm_contacts.csv`.
- Вторичная привязка по телефону из текста сообщения.
- Общий телефон не склеивать.
- Несматч: `pending_attribution`, не создавать нового клиента.
- Идемпотентность по `dedupe_key`.
- Добавить `telegram_history` в `BOT_FORBIDDEN_SOURCE_SYSTEMS`.

## Не входит

- Полная история Telegram/Max.
- Боевая запись без отдельного «да».
- Запись в AMO/Tallanto/CRM.
- Правка YAML.
- ASR или Resolve+Analyze.

## Приёмка

- Залита тестовая manager-only лента.
- Все Telegram-события имеют `allowed_for_bot=0`.
- Повторный запуск не создаёт дублей.
- Есть счётчики: диалоги, сообщения, привязано по телефону, привязано по тексту, pending.
- NEG: общий телефон не склеен; несматч остаётся pending; `allowed_for_bot=1` для `telegram_history` отклоняется; бренд не угадан.
- Отчёт в `tasks/_done/`.
