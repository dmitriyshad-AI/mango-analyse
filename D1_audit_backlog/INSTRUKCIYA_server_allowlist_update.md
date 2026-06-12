# Инструкция: обновление server-side allowlist для AMO note endpoint

Дата: 2026-06-12.

Цель: добавить сделки, в которые черновиковый контур может писать примечания через AI Office API. Клиентские сообщения бот не отправляет: endpoint пишет только внутреннее примечание в AMO.

## Где лежит allowlist

На сервере AI Office рабочий каталог по последнему отчёту: `/opt/ai-office`.

Проверенный кодовый фикс находится в ветке/коммите AI Office `814db41`.

Внутри API есть два уровня допуска, сделка должна быть в обоих:

1. Кодовый жёсткий список:
   `apps/api/app/amo_integration.py`
   переменная `AMO_NOTE_HARD_ALLOWED_LEAD_IDS`.

2. Переменная окружения:
   `CRM_AMO_NOTE_ALLOWED_LEAD_IDS`
   дефолт описан в `apps/api/app/config.py`.

Важно: правка только env недостаточна, если id сделки не входит в `AMO_NOTE_HARD_ALLOWED_LEAD_IDS`.

## Что не трогать

- Не делать `git reset`, `git checkout --`, `git clean`.
- Не деплоить чужие незакоммиченные изменения в `/opt/ai-office`.
- Не менять CRM read-only/MCP контур в том же дереве.
- Не печатать API-ключи и OAuth-токены в терминал, чат или отчёт.
- Не добавлять сделки пачкой без сверки списка Дмитрием.

## Безопасная процедура

1. Зайти на сервер и перейти в каталог:

```bash
cd /opt/ai-office
```

2. Проверить состояние перед правкой:

```bash
git status --short
git rev-parse --short HEAD
grep -n "AMO_NOTE_HARD_ALLOWED_LEAD_IDS" apps/api/app/amo_integration.py
grep -n "CRM_AMO_NOTE_ALLOWED_LEAD_IDS" apps/api/app/config.py
grep -E '^CRM_AMO_NOTE_ALLOWED_LEAD_IDS=' .env
```

Если `git status --short` показывает чужие изменения, не делать `git pull/reset`. Сначала отдельно решить, какие изменения уже должны остаться на сервере.

3. Сделать бэкап файлов, которые будут правиться:

```bash
STAMP=$(date +%Y%m%d_%H%M%S)
cp apps/api/app/amo_integration.py "apps/api/app/amo_integration.py.bak_allowlist_$STAMP"
cp .env ".env.bak_allowlist_$STAMP"
```

4. Добавить нужные lead_id в `AMO_NOTE_HARD_ALLOWED_LEAD_IDS`.

Пример формы:

```python
AMO_NOTE_HARD_ALLOWED_LEAD_IDS = frozenset({49832125, 47854947, 49762441, 49325789})
```

5. Добавить те же lead_id в `.env`:

```bash
CRM_AMO_NOTE_ALLOWED_LEAD_IDS=49832125,47854947,49762441,49325789
```

6. Перезапустить только API-контейнер:

```bash
docker compose up -d --build api
docker compose ps api
```

7. Проверить, что endpoint закрыт без ключа:

```bash
API=https://api.fotonai.online
curl -sS -o /tmp/amo_note_401.json -w '%{http_code}\n' \
  -H "Content-Type: application/json" \
  -d '{"text":"auth probe"}' \
  "$API/api/integrations/amocrm/leads/49832125/notes"
```

Ожидается `401`.

8. Проверить запрет чужой сделки с ключом.

Команда создаёт только отказ, примечание не пишет:

```bash
API=https://api.fotonai.online
curl -sS -o /tmp/amo_note_403.json -w '%{http_code}\n' \
  -H "X-API-Key: $AI_OFFICE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"text":"allowlist negative probe"}' \
  "$API/api/integrations/amocrm/leads/111/notes"
```

Ожидается `403`.

9. Проверка разрешённой сделки пишет реальное примечание в AMO. Запускать только если Дмитрий готов увидеть тестовую запись в карточке:

```bash
API=https://api.fotonai.online
curl -sS -o /tmp/amo_note_200.json -w '%{http_code}\n' \
  -H "X-API-Key: $AI_OFFICE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"text":"Проверка allowlist: тестовое внутреннее примечание."}' \
  "$API/api/integrations/amocrm/leads/47854947/notes"
```

Ожидается `200`, в ответе будет `note_id`. `note_id` — это номер созданного примечания в AMO.

## Если одна сделка вернула 403

Это не должно останавливать весь черновиковый контур Mango. После ТЗ-15 такая пара уходит в карантин `allowlist_desync`, остальные пары продолжают работать.

Причина 403 почти всегда одна из двух:

- сделка есть в локальном Mango allowlist, но не добавлена в server-side allowlist AI Office;
- сделка добавлена в env, но забыта в `AMO_NOTE_HARD_ALLOWED_LEAD_IDS`.

## Постоянный безопасный доступ

Предложение для Дмитрия: завести отдельного пользователя без root-доступа, например `ai-office-deploy`, с SSH-ключом только для деплоя AI Office.

Минимальные права:

- читать `/opt/ai-office`;
- выполнять `git status`, `git fetch`, `git log`;
- через ограниченный sudo запускать только `docker compose up -d --build api` и `docker compose ps api` в `/opt/ai-office`;
- не иметь доступа к shell-history с секретами и не печатать `.env` целиком.

Такой доступ позволит обновлять allowlist и API-код без root и без риска задеть соседние сервисы.
