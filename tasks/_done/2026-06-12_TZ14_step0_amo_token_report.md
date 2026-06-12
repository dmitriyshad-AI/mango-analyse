# TZ-14 Step 0: AMO token lifecycle for `foton-crm-readonly`

Дата: 2026-06-12  
Исполнитель: Codex, сервер `api.fotonai.online` / `api.fotonai.online` backend  
Режим: AMO/Tallanto read-only для MCP-коннектора; записей в AMO/Tallanto не делалось.

## Вывод

Шаг 0 был не закрыт до этой работы: отдельного отчёта `2026-06-12_TZ14_step0_amo_token_report.md` не было, а существующий отчёт `2026-06-12_foton_crm_readonly_mcp_report.md` подтверждал поднятый MCP, но не закрывал конфликт refresh-токенов.

Постановка Claude корректна. Реализован вариант с общим хранилищем токенов:

- основной AMO-контур и `foton-crm-readonly` читают один OAuth-набор из `amo_integration_connections`;
- refresh сериализован файловым замком `/tmp/ai_office_amo_refresh.lock`;
- после ожидания замка код перечитывает connection из БД и повторно проверяет, не обновил ли токен другой процесс;
- авто-refresh из read-only пути теперь делает commit, чтобы новый refresh-token не терялся после закрытия DB-сессии;
- сравнение времени токена устойчиво к значениям БД без timezone.

Причина выбора: минимум новых движущихся частей, не нужны новые AMO client_id/client_secret, основной контур не получает отдельную конкурирующую интеграцию.

## Изменённые файлы на сервере

Серверный репозиторий: `/opt/ai-office`.

- `/opt/ai-office/apps/api/app/amo_integration.py`
- `/opt/ai-office/apps/api/tests/test_amo_integration_unit.py`

Бэкап до правки:

- `/opt/ai-office/backups/tz14_step0_amo_token_20260612_123717/amo_integration.py`
- `/opt/ai-office/backups/tz14_step0_amo_token_20260612_123717/test_amo_integration_unit.py`

Секреты, client_secret, access_token и refresh_token в отчёт не выводились.

## Проверки

Точечные тесты:

- `python3 -m py_compile ...` — PASS.
- `pytest -q tests/test_amo_integration_unit.py` — `10 passed`.
- `pytest -q tests/test_integrations_router_unit.py` из `/workspace_root/apps/api` — `3 passed`.

Полный API pytest:

- `90 passed, 8 failed`.
- Падения не в изменённой токенной логике: старые smoke-ожидания CRM preview routes (`404`), один `review_status`, два router auth `401`, которые отдельно проходят при изолированном запуске.
- Поэтому `formal_pass` для изменённого AMO-token блока есть, но весь AI Office test suite не считается зелёным.

Live smoke через MCP:

- `amo_api_get users` через `foton-crm-readonly` — PASS, `http=200`, `tool_error=False`, users present.

Главный NEG:

- Round 1: refresh основной интеграции — `http=200`, затем MCP `amo_api_get users` — PASS.
- Round 2: refresh основной интеграции — `http=200`, затем MCP `amo_api_get users` — PASS.
- Статус основной интеграции после раундов — `connected=True`, `status=active`, `token_source=oauth`.

Основная интеграция:

- `/integrations/amocrm/status` — PASS.
- read-only `/integrations/amocrm/contacts/by-phone` — PASS, без вывода данных контакта.

WAF / User-Agent:

- Python urllib default — PASS.
- Empty `User-Agent` — PASS.
- Custom `User-Agent` — PASS.

Read-only grep по MCP-слою:

- `apps/api/app/mcp_readonly.py`: запрещённых AMO write methods `POST/PUT/PATCH/DELETE` не найдено.
- `apps/api/app/crm_readonly.py`: запрещённых AMO write methods `POST/PUT/PATCH/DELETE` не найдено.

## Срок жизни токена

Текущий AMO access-token после refresh снова имеет срок около 24 часов:

- `expires_at_utc=2026-06-13T09:47:10.022048+00:00`
- remaining около `23.99` часа на момент проверки.

Ключевое изменение: истечение access-token больше не требует ручной переавторизации при нормальном OAuth-refresh; read-only путь обновляет токены через общий сериализованный refresh и сохраняет новый refresh-token.

## Semantic Review

Verdict: `PASS_WITH_NOTES`.

Что прошло:

- выбранный вариант соответствует бизнес-цели: не блокировать AMO read-only выгрузки из-за суточного access-token;
- не добавлены новые write-инструменты в MCP;
- live-проверки показывают, что основной refresh не отзывает работоспособность коннектора;
- WAF больше не выглядит блокером для Python-клиентов.

Остаточные риски:

- весь API test suite AI Office не зелёный из-за несвязанных падений; это не блокирует сам token-flow, но требует отдельной уборки;
- серверный git tree уже был грязным до правки, поэтому эти изменения не зафиксированы отдельным коммитом, чтобы не смешивать чужие изменения в тех же файлах.

## Решение для D4

Шаг 0 AMO-token lifecycle для `foton-crm-readonly` закрыт по обязательным live-проверкам. D4 может переходить к AMO read-only части ТЗ-14, соблюдая лимиты: постранично, `limit=50`, пауза не меньше 0.5 с, без записей в AMO/Tallanto.
