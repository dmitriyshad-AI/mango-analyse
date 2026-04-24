# Точки интеграции с Mango analyse

## Что уже встроено

- runtime-код: `src/mango_mvp/amocrm_runtime/`
- launcher: `stable_runtime/run-amocrm-runtime.sh`
- env templates: `stable_runtime/amocrm_runtime/.env.*.example`
- deal-analysis слой: `src/mango_mvp/amocrm_runtime/deals.py`
- deal dossier builder: `src/mango_mvp/amocrm_runtime/deal_dossier.py`
- LLM deal analyzer: `src/mango_mvp/amocrm_runtime/deal_llm.py`
- загрузка истории по телефону из Mango analyse: `src/mango_mvp/amocrm_runtime/phone_context.py`

## Как это связывается с основным проектом

### 1. Источник истории общения

Deal-analysis берет историю клиента из последнего актуального sales export:

- `stable_runtime/sales_master_export*/master_contacts_ru.csv`
- `stable_runtime/sales_master_export*/master_calls_ru.csv`

Автопоиск идет по самой свежей директории, где есть оба CSV.

### 2. Контактный и сделочный контур amoCRM

Для amo runtime теперь есть два слоя:

- `src/mango_mvp/amocrm_runtime/amo_integration.py`
  - OAuth/shared DB access context
  - чтение и обновление contact custom fields
  - чтение и обновление lead custom fields
  - чтение contacts / leads / notes / tasks / pipelines / users

- `src/mango_mvp/amocrm_runtime/deals.py`
  - phone -> contact -> best lead candidate
  - heuristic screening
  - shadow/primary selection
  - write-back guardrails
  - queue builder по недавно закрытым сделкам

- `src/mango_mvp/amocrm_runtime/deal_dossier.py`
  - сбор полного dossier сделки
  - history summaries
  - transcript context из source DB
  - notes/tasks normalization

- `src/mango_mvp/amocrm_runtime/deal_llm.py`
  - strict JSON verdict
  - codex_cli runtime с локальным writable `CODEX_HOME`
  - LLM cache

### 3. Live-слой Tallanto

Теперь live-доступ к CRM Tallanto встроен как read-only слой:

- `src/mango_mvp/amocrm_runtime/tallanto_api.py`
  - HTTP-клиент Tallanto API
  - schema discovery
  - поиск контакта по телефону
  - получение контекста по `contact_id`

- `src/mango_mvp/amocrm_runtime/tallanto_export.py`
  - schema export модулей, полей и enum values

- `src/mango_mvp/amocrm_runtime/tallanto_context.py`
  - компактный live-context для dossier сделки
  - приоритет `ID Tallanto` из master table
  - fallback на поиск по телефону

- `src/mango_mvp/amocrm_runtime/routers/tallanto.py`
  - runtime endpoints для health/schema/contact/context

- `src/mango_mvp/amocrm_runtime/phone_context.py`
  - читает `ID Tallanto` и `Статус матчинга Tallanto` из master contacts

- `src/mango_mvp/amocrm_runtime/deal_dossier.py`
  - автоматически подмешивает `tallanto_live` в dossier сделки

### 4. Куда писать результаты анализа сделок

Результат должен жить в двух местах:

- source of truth: файловые очереди и анализ в Mango analyse
- витрина: custom fields сделки в amoCRM

Файловая очередь пишется в:

- `stable_runtime/amocrm_runtime/deal_analysis/`

### 5. Какие endpoints уже готовы

- `/api/integrations/amocrm/contacts/by-phone`
- `/api/integrations/amocrm/leads/by-phone`
- `/api/integrations/amocrm/deals/dossier-by-phone`
- `/api/integrations/amocrm/deals/analyze-by-phone`
- `/api/integrations/amocrm/deals/writeback`
- `/api/integrations/amocrm/deals/queue/build`
- `/api/integrations/amocrm/deals/queue/latest`
- `/api/integrations/amocrm/lead-fields/sync`
- `/api/integrations/tallanto/health`
- `/api/integrations/tallanto/modules`
- `/api/integrations/tallanto/fields`
- `/api/integrations/tallanto/schema/export`
- `/api/integrations/tallanto/contact/by-phone`
- `/api/integrations/tallanto/opportunities/by-contact`
- `/api/integrations/tallanto/context/by-phone`
- `/api/integrations/tallanto/context/by-contact-id`

### 6. Что еще не реализовано

- отдельный write-layer для reopen/restage сделки
- controlled reopen сделки по verdict `reopen_recommended`
- автоматическое создание lead fields не делаем: поля создаются только вручную
- write-back в Tallanto не делаем: контур строго read-only
