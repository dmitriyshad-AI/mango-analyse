# amoCRM runtime для Mango analyse

Этот runtime поднимается отдельно от основного UI/ASR пайплайна и переиспользует handoff-код из `prod_runtime_transfer`.

## Что уже умеет

- `GET/POST /api/integrations/amocrm/secrets`
- `GET /api/integrations/amocrm/callback`
- `GET /api/integrations/amocrm/status`
- `POST /api/integrations/amocrm/refresh`
- `POST /api/integrations/amocrm/contact-fields/sync`
- `POST /api/integrations/amocrm/lead-fields/sync`
- `GET /api/integrations/amocrm/contacts/by-phone`
- `GET /api/integrations/amocrm/leads/by-phone`
- `POST /api/integrations/amocrm/deals/dossier-by-phone`
- `POST /api/integrations/amocrm/deals/analyze-by-phone`
- `POST /api/integrations/amocrm/deals/writeback`
- `POST /api/integrations/amocrm/deals/queue/build`
- `GET /api/integrations/amocrm/deals/queue/latest`
- `GET /api/integrations/tallanto/health`
- `GET /api/integrations/tallanto/modules`
- `GET /api/integrations/tallanto/fields`
- `POST /api/integrations/tallanto/schema/export`
- `GET /api/integrations/tallanto/contact/by-phone`
- `GET /api/integrations/tallanto/opportunities/by-contact`
- `GET /api/integrations/tallanto/context/by-phone`
- `GET /api/integrations/tallanto/context/by-contact-id`
- `GET /health`

## Рекомендуемый режим в этом проекте

Используйте `shared DB mode`.

Причина:
- в общей Postgres БД уже живет рабочая OAuth state `amo_integration_connections`
- runtime умеет автоматически брать живой access context и refresh-ить токены
- `direct token` оставлен только как fallback

## Как запустить

Если runtime работает не прямо на VPS, сначала поднимите shared DB tunnel:

```bash
cd "/Users/dmitrijfabarisov/Projects/Mango analyse"
./stable_runtime/start-amocrm-shared-db-tunnel.sh
```

Потом в отдельном терминале:

```bash
cd "/Users/dmitrijfabarisov/Projects/Mango analyse"
./stable_runtime/run-amocrm-runtime.sh
```

После старта runtime будет слушать:

- `http://127.0.0.1:8010/health`
- `http://127.0.0.1:8010/api/integrations/amocrm/status`

## Откуда берутся env

Приоритет:

1. `stable_runtime/amocrm_runtime/.env.private`
2. `prod_runtime_transfer/.env.private`

## Быстрый smoke-check

```bash
curl http://127.0.0.1:8010/health
curl -H 'X-API-Key: <KEY>' http://127.0.0.1:8010/api/integrations/amocrm/status
curl -X POST -H 'X-API-Key: <KEY>' http://127.0.0.1:8010/api/integrations/amocrm/refresh
curl -X POST -H 'X-API-Key: <KEY>' http://127.0.0.1:8010/api/integrations/amocrm/contact-fields/sync
curl -X POST -H 'X-API-Key: <KEY>' http://127.0.0.1:8010/api/integrations/amocrm/lead-fields/sync
curl -H 'X-API-Key: <KEY>' 'http://127.0.0.1:8010/api/integrations/amocrm/leads/by-phone?phone=%2B79154476613'
curl -X POST -H 'X-API-Key: <KEY>' -H 'Content-Type: application/json' \
  -d '{"phone":"+79154476613"}' \
  http://127.0.0.1:8010/api/integrations/amocrm/deals/dossier-by-phone
curl -X POST -H 'X-API-Key: <KEY>' -H 'Content-Type: application/json' \
  -d '{"phone":"+79154476613"}' \
  http://127.0.0.1:8010/api/integrations/amocrm/deals/analyze-by-phone
curl -H 'X-API-Key: <KEY>' http://127.0.0.1:8010/api/integrations/tallanto/health
curl -H 'X-API-Key: <KEY>' 'http://127.0.0.1:8010/api/integrations/tallanto/contact/by-phone?phone=%2B79012900755'
curl -H 'X-API-Key: <KEY>' 'http://127.0.0.1:8010/api/integrations/tallanto/context/by-contact-id?contact_id=<TALLANTO_ID>'
```

## Что делает deal-analysis shadow MVP

1. Берет телефон из звонка.
2. Ищет contact в amoCRM.
3. Находит связанные leads/deals.
4. Выбирает лучшую сделку по pipeline, статусу, времени, менеджеру и активности.
5. Собирает `deal dossier`:
   - большая рабочая таблица по контакту
   - хронология общения
   - история звонков по телефону
   - transcript context по последним звонкам
   - notes/tasks по сделке
   - live Tallanto context по ученику, сделкам, заявкам, группам и финансам
6. Прогоняет:
   - heuristic screening
   - LLM verdict в `shadow mode`
7. Выставляет:
   - `AI-вердикт по закрытию`
   - `AI-risk: premature close`
   - `AI-основание вердикта`
   - `AI-рекомендованный следующий шаг`
   - `AI-дата следующего касания`
   - `AI-сводка по сделке`
8. В `shadow mode` ничего автоматически не пишет в amoCRM.
9. Controlled write-back пишет только в сервисные поля сделки и не меняет статус сделки.
10. Может собрать файловую очередь кандидатов на возврат в работу.

## Ограничения MVP

- Матчинг идет по схеме `phone -> contact -> leads`, а не по прямой связи звонка со сделкой.
- По умолчанию runtime работает в `llm_shadow`, поэтому `/deals/writeback` блокируется guardrails.
- `AI office` поле сейчас сознательно не пишется автоматически.
- По умолчанию queue builder анализирует только недавно закрытые потерянные сделки.
- Tallanto работает в режиме read-only и не пишет обратно в CRM.
- Если в master-слое уже есть точный `ID Tallanto`, dossier сначала идет по нему, а только потом падает назад на поиск по телефону.
