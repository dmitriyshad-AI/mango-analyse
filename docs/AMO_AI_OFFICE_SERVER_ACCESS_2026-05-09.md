# amoCRM access through AI Office API

Дата: 2026-05-09

## Решение

Mango Analyse не должен использовать прямой `AMOCRM_ACCESS_TOKEN` для production-доступа к amoCRM.
Production-доступ к amoCRM идет через AI Office API:

- `AI_OFFICE_API_BASE_URL=https://api.fotonai.online`
- `AMOCRM_STATUS_URL=https://api.fotonai.online/api/integrations/amocrm/status`
- `AMOCRM_CONTACT_FIELDS_SYNC_URL=https://api.fotonai.online/api/integrations/amocrm/contact-fields/sync`

`AI_OFFICE_API_KEY` хранится в локальном `.env` и не должен попадать в примерные env-файлы или документацию.

## Почему так

AI Office держит OAuth-состояние amoCRM на сервере:

- access token;
- refresh token;
- refresh flow;
- актуальный статус подключения;
- каталог полей amoCRM.

Mango Analyse передает запросы в AI Office API с `X-API-Key`.
Поэтому локальный `AMOCRM_ACCESS_TOKEN` в Mango не является production-источником правды и может быть пустым или устаревшим.

## Read-only проверка

```bash
cd "/Users/dmitrijfabarisov/Projects/Mango analyse"
set -a
source .env
set +a

curl -sS \
  -H "X-API-Key: $AI_OFFICE_API_KEY" \
  "$AMOCRM_STATUS_URL" | jq
```

Ожидаемые ключевые поля:

```json
{
  "connected": true,
  "status": "active",
  "account_base_url": "https://educent.amocrm.ru",
  "token_source": "oauth"
}
```

## Safety

Эта проверка только читает статус интеграции. Она не пишет в amoCRM, не меняет сделки, контакты, задачи или поля.
