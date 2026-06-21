> DONE 2026-06-21 13:56 | ветка codex/wappi-history | codex

> TAKE 2026-06-21 13:27 | ветка codex/wappi-history | codex

Ветка: codex/wappi-history
Зоны: scripts/, src/mango_mvp/customer_timeline/, src/mango_mvp/integrations/, tests/, tasks/, docs/worktrees_registry.md, audits/_inbox/, product_data/customer_timeline/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest --collect-only -q
Семантический-аудит: да

# ТЗ — выкачка истории мессенджеров через Wappi в тестовую память

Источник: `/Users/dmitrijfabarisov/Claude Projects/Foton/2026-06-21_TZ_Wappi_vykachka_messendzherov.md`.

## Цель

Выкачать историю Telegram/Max через Wappi по 4 профилям и загрузить в тестовую копию `customer_timeline` как manager-only историю клиента.

## Жёсткие правила

- Все Wappi-вызовы только через `DefaultDenyTransport`.
- Любой не-GET должен давать `TransportDenied`.
- Прямой `_json_http_request` / `transport=None` запрещён для Wappi.
- Токены только из `~/.mango_secrets`, не писать в git/отчёт.
- Полные телефоны и ПДн в отчёты не выводить.
- AMO/Tallanto/CRM/live/stable_runtime/YAML не трогать.
- Только тестовая копия памяти.

## Профили

- `ec2eed50-b55f`: Фотон TG.
- `18b255b8-7a67`: УНПК TG.
- `2952990f-9e4c`: Фотон Max.
- `152b441d-81a2`: УНПК Max.

Бренд брать только из профиля, не угадывать.

## Импорт

- `source_system`: `wappi_telegram` / `wappi_max`.
- `allowed_for_bot=0`.
- Привязка не по телефону: `profile_id + chat_id -> lead_id/contact_id/customer_id`.
- Статические `draft_loop_pairs` валидны.
- Нет уверенного резолва: `pending_attribution`, клиента не создавать.
- Идемпотентность по message id / sha256.

## Приёмка

- По 4 профилям есть счётчики чатов/сообщений.
- Бренд соответствует профилю.
- Повторный прогон даёт 0 новых строк.
- `allowed_for_bot=1` отклоняется.
- 0 отправленных сообщений, все Wappi-вызовы через `DefaultDenyTransport`.
- Отчёт в `tasks/_done/`, audit pack в `audits/_inbox/`.
