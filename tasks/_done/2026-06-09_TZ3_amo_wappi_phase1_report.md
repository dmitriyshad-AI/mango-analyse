# TZ3 AMO + Wappi phase 1 report

Дата: 2026-06-09
Ветка: main

## Read-only карта

- Готового Wappi-клиента в коде не было.
- AMO-код в проекте уже есть, но связан со старым runtime/writeback; для пилотного транспорта безопаснее сделан отдельный изолированный scaffold.
- Фаза 1 допускает одну write-операцию: внутреннее AMO-примечание только в тестовую сделку из allowlist. Вся остальная логика — read-only.
- Wappi profile list по документации/audit: `GET /tapi/profile/all/get` для Telegram и `GET /maxapi/profile/all/get` для MAX, с `Authorization`.

## Что изменено

Коммит: `ef05378f Add AMO Wappi phase1 draft note scaffold`

- Добавлен модуль `mango_mvp.integrations.amo_wappi_phase1`.
- Секреты читаются из env или внешнего env-файла `~/.mango_secrets/amo_wappi.env`; секретов в репозитории нет.
- Добавлен пример config без секретов: `config/amo_wappi_phase1.example.json`.
- Добавлен README: `docs/AMO_WAPPI_PHASE1_README_2026-06-09.md`.
- AMO client:
  - read-only: pipelines, lead, contacts, contact;
  - write: `add_draft_note_to_test_lead`, строго после allowlist-проверки `allowed_test_lead_ids`.
- Wappi client:
  - read-only список профилей Telegram/MAX;
  - profile_id нормализуется и дальше мапится на brand через config-файл, не хардкодом.
- Draft note содержит маркер `ЧЕРНОВИК БОТА, не отправлено`, бренд, время, profile_id и текст черновика.
- Добавлена структура JSONL-журнала правок менеджера: `bot_draft_text` рядом с `manager_sent_text` и `reason_codes`.

## NEG и проверки

- Запись в AMO note вне `allowed_test_lead_ids` блокируется до HTTP-вызова.
- Config неизвестного `profile_id` fail-closed.
- Неизвестный бренд для draft note запрещён.
- Wappi profile list использует только read-only endpoints.
- AMO read methods строят ожидаемые GET-запросы.
- Draft note в тестовую сделку содержит верный маркер и бренд.
- Manager edit log сохраняет предложенный текст бота и отправленный текст менеджера рядом.
- Проверен diff на отсутствие переданных Wappi-секретов.

## Тесты

- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_amo_wappi_phase1.py`
  - `8 passed`
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests`
  - `2848 passed, 2 skipped, 1 warning`

## Что не выполнялось live

- Реальные AMO/Wappi вызовы не запускались.
- В AMO note ничего не записывалось: для этого нужен заполненный внешний config с `allowed_test_lead_ids` и явный запуск на тестовой сделке.
- Wappi login/password сохранены вне репозитория в env-файл, но для read-only API profile list, вероятно, нужен Wappi API token (`WAPPI_TELEGRAM_TOKEN`/`WAPPI_MAX_TOKEN` или общий `WAPPI_API_TOKEN`).
