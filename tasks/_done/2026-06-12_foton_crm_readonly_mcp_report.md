# foton-crm-readonly MCP report, 2026-06-12

## Итог

Поднят отдельный read-only MCP-контур для чтения AMO CRM и Tallanto через действующий сервер AI Office.

Рабочий URL коннектора:

```text
https://api.fotonai.online/api/mcp/foton-crm-readonly
```

Токен не записан в git, не включён в отчёт и не выводился в чат. Он сохранён только:

- на сервере в `/opt/ai-office/.env`;
- в приватном локальном файле Дмитрия `~/.mango_secrets/foton_crm_readonly_mcp_connector.env` с правами `600`.

Важно: домен `api.foton.online` сейчас не указывает на этот сервер и не проходит TLS для этого сервиса. Проверка дала TLS SNI error. Для точного домена из ТЗ нужен отдельный DNS/сертификатный шаг. Рабочий публичный домен текущего контура: `api.fotonai.online`.

## Что поднято

Маршрут MCP:

- `GET /api/mcp/foton-crm-readonly` — служебная информация, требует Bearer token.
- `POST /api/mcp/foton-crm-readonly` — JSON-RPC MCP methods: `initialize`, `tools/list`, `tools/call`.

Инструменты v1:

- `amo_find_contact(phone)`
- `amo_get_lead(lead_id)`
- `amo_recent_leads(since_iso, limit <= 100)`
- `amo_api_get(path, params, limit <= 250)`
- `tallanto_context(phone, tallanto_id = null)`
- `tallanto_fields(module)`
- `tallanto_select(module, filters, limit <= 50)`

AMO `amo_api_get` пропускает только read-only GET по whitelist:

- `contacts`
- `leads`
- `companies`
- `tasks`
- `notes` только как entity notes, например `leads/{id}/notes`
- `users`
- `pipelines` как alias на `leads/pipelines`
- `custom_fields`
- `events`

Tallanto whitelist:

- `Contact`
- `Opportunity`
- `Request`
- `most_finances`
- `most_abonements`
- `most_courses`
- `most_class`
- `most_template_abonements`
- `CoursesContactsRelationship`
- `ClassContactsRelationship`

## Ограничения и безопасность

- Авторизация: Bearer token из серверного конфига.
- Лимит: 60 запросов в минуту на токен.
- Таймаут инструмента: 10 секунд.
- Лимит ответа: 200000 байт, дальше обрезка с маркером `truncated=true`.
- Журнал вызовов: `/opt/ai-office/runtime/foton_crm_readonly_mcp/calls.jsonl`.
- В журнал пишутся время, инструмент, статус, fingerprint токена и маскированные аргументы.
- Фоновых синков и кэшей ПДн не добавлено.
- Сервис изолирован в AI Office API и не связан с боевым ботом.
- Записывающих MCP-инструментов нет.

## Изменённые файлы

На сервере `/opt/ai-office`:

- `apps/api/app/mcp_readonly.py`
- `apps/api/app/tallanto_api.py`
- `apps/api/app/tallanto_context.py`
- `apps/api/app/config.py`
- `apps/api/app/main.py`
- `docker-compose.yml`
- `.env` — только секреты/лимиты, значения в отчёт не включены

Локально в `/Users/dmitrijfabarisov/Projects/AI Office`:

- `apps/api/app/mcp_readonly.py`
- `apps/api/app/tallanto_api.py`
- `apps/api/app/tallanto_context.py`
- `apps/api/app/config.py`
- `apps/api/app/main.py`
- `docker-compose.yml`
- `apps/api/tests/test_mcp_readonly_unit.py`

Рабочее дерево AI Office было грязным до начала задачи; чужие изменения не откатывались и не смешивались в коммит.

## Проверки

Локально:

```text
PYTHONDONTWRITEBYTECODE=1 apps/api/.venv/bin/python -m pytest -q \
  apps/api/tests/test_mcp_readonly_unit.py \
  apps/api/tests/test_tallanto_api_unit.py \
  apps/api/tests/test_main.py \
  apps/api/tests/test_integrations_router_unit.py

15 passed, 1 warning
```

```text
docker compose config --quiet
compose_ok
```

На сервере:

- `docker compose config --quiet` -> `compose_ok`
- `python3 -m py_compile apps/api/app/mcp_readonly.py apps/api/app/tallanto_api.py apps/api/app/tallanto_context.py apps/api/app/config.py apps/api/app/main.py` -> ok
- `docker compose up -d --build api` -> API контейнер поднят
- `GET https://api.fotonai.online/api/health` -> `status: ok`
- вызов MCP без токена -> `401 Bearer token is required`
- `tools/list` с токеном -> 7 tools
- `tallanto_fields("Contact")` -> 84 поля
- `amo_api_get("users", limit=1)` -> AMO read ok
- `amo_api_get("pipelines", limit=10)` -> 3 pipeline records in embedded payload
- `amo_recent_leads("2026-06-01T00:00:00+03:00", limit=1)` -> 1 lead
- `amo_get_lead(lead_id)` по найденной сделке -> id/status читаются
- `amo_find_contact("+70000000000")` -> пустой список, не ошибка
- `tallanto_context("+70000000000")` -> пустой список контактов, не ошибка
- `tallanto_select("ForbiddenModule")` -> controlled error
- `amo_api_get("account")` -> controlled whitelist error

Read-only grep по MCP runtime:

```text
grep -RInE 'method\s*=\s*["'\''](POST|PUT|PATCH|DELETE)["'\'']|\.put\(|\.patch\(|\.delete\(|requests\.(post|put|patch|delete)|httpx\.(post|put|patch|delete)|set_entry|delete_entry|create_lead_common_note|send_contact_custom_field_update|send_lead_custom_field_update|refresh_connection_tokens|exchange_callback_code|record_external_secrets' \
  apps/api/app/mcp_readonly.py apps/api/app/tallanto_context.py
```

Результат: совпадений нет.

## Инструкция подключения

Claude Desktop:

1. `Settings`
2. `Connectors`
3. `Add custom connector`
4. URL: `https://api.fotonai.online/api/mcp/foton-crm-readonly`
5. Token: взять из приватного файла Дмитрия, не копировать в git/отчёты/чат.

Smoke после подключения:

- `tallanto_fields("Contact")` должен вернуть список полей.
- `amo_find_contact(тестовый номер)` должен вернуть найденные контакты или пустой список без ошибки.

## Остаточные вопросы

1. Для адреса `api.foton.online` нужен DNS/сертификатный шаг: сейчас этот домен не приходит на текущий сервер AI Office.
2. Коммит не делал: локальный и серверный AI Office уже содержали много несвязанных изменений. Для коммита нужно отдельно отделить MCP-пакет от существующей грязи.
3. Запись в AMO/Tallanto по-прежнему запрещена; этот сервис не содержит write-инструментов и не должен их получать ни под каким флагом.
