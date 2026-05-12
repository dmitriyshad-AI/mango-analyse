# Unified Customer Timeline Architecture

Дата: 2026-05-12
Ветка: SaaS/productization
Статус: architecture plan before implementation

## Зачем нужен слой

Unified Customer Timeline должен стать общей лентой клиента: звонки Mango,
письма, вложения, AMO, Tallanto, Telegram/Max/Web-chat и будущие действия бота
сводятся в одну понятную историю.

Главная цель на текущем этапе: польза для отдела продаж и РОПа. Слой должен
показывать, что с клиентом происходило, кто общался, что обещали, что клиент
спрашивал, какие есть риски и какое следующее действие разумно сделать.

Этот слой не заменяет исходные системы. Он строит read model поверх локальных
снимков, архивов и результатов обработки.

## Границы ответственности

### Этот диалог

- архитектура единой ленты клиента;
- product DB/read model;
- read-only API;
- dashboard/workspace;
- bot context builder;
- интеграция готовых артефактов из других потоков.

### Processing-диалог

- ASR;
- Resolve + Analyze;
- качество транскриптов;
- CRM-ready/writeback gates;
- обработка звонков и их аналитика.

### Mail-диалог

- IMAP read-only ingest;
- raw `.eml`;
- вложения;
- mail archive DB;
- email identity map;
- matching report `email -> Tallanto/AMO/Mango`.

Unified Customer Timeline принимает результаты этих потоков, но не запускает
тяжелую обработку звонков и не пишет в live CRM.

## Принцип хранения

Рекомендуемая схема для client-hosted модели:

- SQLite хранит метаданные, связи, индексы, статусы и read model;
- тяжелые файлы хранятся на диске;
- оригиналы не перетираются;
- каждый файл получает `sha256`, размер, MIME/type, source ref;
- каждый импорт идемпотентен: повторный запуск не создает дубли.

Пример структуры product root:

```text
product_data/
  db/
    mango_product.sqlite
    channel_runtime.sqlite
    customer_timeline.sqlite
  raw/
    mango/YYYY/MM/DD/{call_id}/audio.*
    mango/YYYY/MM/DD/{call_id}/raw.json
    mail/YYYY/MM/DD/{message_key}/raw.eml
    amo/YYYY/MM/DD/{entity_id}.json
    tallanto/YYYY/MM/DD/{student_id}.json
  files/
    sha256/ab/cd/{hash}.{ext}
  processed/
    transcripts/{call_id}.json
    mail_text/{message_key}.txt
    attachment_text/{artifact_id}.txt
    analysis/{event_id}.json
  reports/
    timeline_readiness/
    identity_conflicts/
  backups/
```

На раннем этапе `customer_timeline.sqlite` лучше держать отдельной SQLite DB.
Так мы не ломаем существующий `product_db` и можем безопасно заменить схему,
если первая версия окажется недостаточной.

## Основные сущности

### 1. Customer Identity

Каноническая запись клиента или семьи.

Поля:

- `customer_id`;
- `tenant_id`;
- `display_name`;
- `identity_status`: `strong`, `partial`, `ambiguous`, `unmatched`;
- `first_seen_at`;
- `last_seen_at`;
- `touch_count`;
- `primary_phone`;
- `primary_email`;
- `summary_json`;
- `created_at`;
- `updated_at`.

Важно: один телефон или email может вести к нескольким ученикам/родителям.
Это не должно автоматически склеиваться в одного клиента без статуса конфликта.

### 2. Identity Link

Связь клиента с конкретным идентификатором.

Типы:

- `phone`;
- `email`;
- `amo_contact_id`;
- `amo_lead_id`;
- `tallanto_student_id`;
- `tallanto_parent_ref`;
- `mango_client_phone`;
- `telegram_user_id`;
- `max_user_id`;
- `web_chat_user_id`.

Поля:

- `link_id`;
- `customer_id`;
- `link_type`;
- `link_value`;
- `source_system`;
- `source_ref`;
- `confidence`;
- `match_class`: `strong_unique`, `duplicate`, `ambiguous`, `inferred`, `manual`;
- `evidence_json`;
- `first_seen_at`;
- `last_seen_at`.

### 3. Opportunity / Thread

Отдельная цепочка внутри клиента: сделка, курс, ребенок, абонемент, запрос,
возврат, сервисный вопрос.

Зачем нужно: один родитель может иметь несколько детей, несколько курсов и
несколько сделок. Лента клиента должна быть общей, но продажи часто нужно
анализировать по отдельной возможности.

Поля:

- `opportunity_id`;
- `customer_id`;
- `opportunity_type`: `amo_deal`, `tallanto_course`, `service_case`, `renewal`, `unknown`;
- `source_system`;
- `source_id`;
- `title`;
- `status`;
- `product_context`;
- `opened_at`;
- `closed_at`;
- `confidence`;
- `evidence_json`.

### 4. Timeline Event

Одна запись в единой ленте.

Типы событий:

- `mango_call`;
- `call_transcript`;
- `call_analysis`;
- `email_message`;
- `email_attachment`;
- `amo_contact_snapshot`;
- `amo_deal_stage`;
- `amo_note`;
- `amo_task`;
- `tallanto_student_snapshot`;
- `tallanto_payment`;
- `tallanto_abonement`;
- `tallanto_group`;
- `channel_message`;
- `bot_draft`;
- `manager_action`;
- `system_note`.

Поля:

- `event_id`;
- `customer_id`;
- `opportunity_id`;
- `tenant_id`;
- `event_type`;
- `event_at`;
- `source_system`;
- `source_id`;
- `direction`: `inbound`, `outbound`, `internal`, `system`;
- `actor_name`;
- `actor_ref`;
- `subject`;
- `text_preview`;
- `summary`;
- `stage_before`;
- `stage_after`;
- `importance`;
- `match_status`;
- `confidence`;
- `record_json`;
- `created_at`.

### 5. Event Artifact

Файл или текстовый слой, связанный с событием.

Типы:

- `call_audio`;
- `call_transcript_json`;
- `raw_email_eml`;
- `mail_attachment`;
- `attachment_text`;
- `api_raw_json`;
- `analysis_json`;
- `report_file`.

Поля:

- `artifact_id`;
- `event_id`;
- `artifact_type`;
- `path`;
- `sha256`;
- `size_bytes`;
- `mime_type`;
- `source_system`;
- `source_ref`;
- `extraction_status`;
- `created_at`.

### 6. Derived Signal

Вывод системы из одного или нескольких событий.

Примеры:

- клиент спросил цену;
- клиент сомневается;
- обещали отправить материалы;
- клиент оплатил;
- клиент уже учится;
- просит возврат;
- нужна задача менеджеру;
- бот может ответить сам;
- требуется человек.

Поля:

- `signal_id`;
- `customer_id`;
- `opportunity_id`;
- `event_id`;
- `signal_type`;
- `severity`;
- `confidence`;
- `evidence_text`;
- `recommended_action`;
- `requires_manager_review`;
- `created_at`.

### 7. Bot Context Chunk

Подготовленный кусок знания для будущего бота.

Поля:

- `chunk_id`;
- `customer_id`;
- `opportunity_id`;
- `event_id`;
- `source_system`;
- `chunk_type`;
- `text`;
- `summary`;
- `event_at`;
- `freshness_score`;
- `relevance_tags`;
- `allowed_for_bot`;
- `requires_manager_review`;
- `created_at`.

На первом этапе достаточно SQLite FTS5. Векторный поиск можно добавить позже
отдельно, когда появятся реальные сценарии бота.

## Слои архитектуры

### Layer A. Source Adapters

Адаптеры превращают внешние данные в единый формат, но не принимают решения.

- Mango call adapter: из `product_calls`, транскриптов и analysis JSON.
- Mail adapter: из mail archive DB/JSONL от почтового диалога.
- AMO adapter: из read-only AMO snapshot и writeback/readback reports.
- Tallanto adapter: из Tallanto export/API snapshot.
- Channel adapter: из `src/mango_mvp/channels`.

### Layer B. Identity Resolver

Строит связи:

```text
email -> Tallanto student/parent -> phone -> Mango calls -> AMO contact/deal
phone -> AMO/Tallanto -> customer
channel user -> phone/email/contact if known
```

Результат не обязан быть идеальным. Главное, чтобы система явно различала:

- уверенно найдено;
- найдено несколько кандидатов;
- не найдено;
- конфликт между источниками.

### Layer C. Timeline Event Store

Идемпотентно сохраняет канонические события. Уникальность события считается по:

```text
tenant_id + source_system + source_id + event_type
```

Повторный импорт должен обновить существующее событие или пропустить его как
дубликат.

### Layer D. Timeline Builder

Сортирует события по времени, группирует по клиенту и opportunity/thread,
добавляет краткие summary, touch counts, последние действия, незакрытые риски.

### Layer E. Sales State / Insight Layer

Выводит скрытые стадии:

- `new_request`;
- `discovery`;
- `offer_explained`;
- `price_discussion`;
- `objection_handling`;
- `materials_sent`;
- `decision_wait`;
- `payment_intent`;
- `paid_or_enrolled`;
- `existing_client_service`;
- `reactivation`;
- `lost_or_stalled`.

### Layer F. Read-only API

Первый набор endpoint-ов:

- `GET /customers`;
- `GET /customers/{customer_id}`;
- `GET /customers/{customer_id}/timeline`;
- `GET /customers/{customer_id}/signals`;
- `GET /customers/{customer_id}/bot-context`;
- `GET /timeline/search?q=...`;
- `GET /timeline/readiness`.

На первом этапе это может быть Python facade без HTTP, по аналогии с
`ProductApiFacade`. HTTP слой добавляется после стабилизации contracts.

### Layer G. Dashboard / Approval Workspace

Панели:

- список клиентов;
- лента клиента;
- identity conflicts;
- последние важные события;
- письма без сопоставления;
- звонки без клиента;
- suggested next actions;
- bot context preview.

### Layer H. Bot Context Builder

Готовит компактный контекст для будущего бота:

- кто клиент;
- последние события;
- активная сделка/курс;
- что уже обещали;
- что нельзя обещать;
- какие вопросы уже обсуждались;
- какие источники подтверждают вывод.

Бот не должен читать raw archive напрямую.

## SQLite схема MVP

Минимальные таблицы:

```text
timeline_schema_migrations
customer_identities
identity_links
customer_opportunities
timeline_events
event_artifacts
derived_signals
bot_context_chunks
timeline_import_runs
timeline_conflicts
```

Индексы:

```text
identity_links(link_type, link_value)
timeline_events(customer_id, event_at)
timeline_events(source_system, source_id, event_type)
timeline_events(event_type, event_at)
event_artifacts(sha256)
derived_signals(customer_id, signal_type)
bot_context_chunks(customer_id, event_at)
```

FTS5:

```text
timeline_event_fts(event_id, subject, text_preview, summary)
bot_context_fts(chunk_id, text, summary)
```

## Входные артефакты

### Уже есть или создаются

- product DB звонков и capture lifecycle;
- channel persistence DB;
- read-only AMO snapshot exporter;
- Tallanto export `Ученики.csv`;
- read-only IMAP snapshot;
- mail archive files from отдельного mail-диалога;
- processing reports/transcripts/analysis from processing-диалога.

### Что нужно получить от mail-диалога

- `mail_archive.sqlite`;
- raw `.eml` archive root;
- attachments manifest;
- extracted text manifest;
- `email_identity_map.sqlite` or JSONL;
- `mail_matching_report.json`;
- class counts: matched, ambiguous, unmatched, internal/system.

### Что нужно получить от processing-диалога

- список обработанных звонков;
- transcript refs;
- analysis refs;
- client phone;
- AMO entity refs if resolved;
- quality flags;
- ROP/CRM-ready outcome.

## План реализации

### Фаза 1. Architecture Contracts

Сделать:

- `docs/UNIFIED_CUSTOMER_TIMELINE_ARCHITECTURE_2026-05-12.md`;
- contracts/dataclasses для customer identity, identity link, timeline event,
  artifact, signal, bot context chunk;
- тесты валидации contracts;
- safety contract: no live CRM write, no ASR/R+A, no source deletion.

Definition of done:

- contracts сериализуются в JSON;
- есть stable id/idempotency key;
- тесты покрывают validation/idempotency/no live side effects.

### Фаза 2. Timeline SQLite Store

Сделать:

- отдельный `customer_timeline.sqlite`;
- миграции;
- WAL;
- repository/store;
- idempotent upsert;
- FTS5, если доступен в локальном SQLite;
- fallback без FTS5, если сборка SQLite не поддерживает FTS5.

Definition of done:

- повторный импорт не создает дубли;
- source ref и sha256 сохраняются;
- read-only open mode работает;
- stable_runtime paths блокируются.

### Фаза 3. Identity Map Import

Сделать:

- импорт Tallanto identity map;
- импорт AMO snapshot identities;
- нормализация email/phone;
- identity conflict report;
- классы `strong_unique`, `ambiguous`, `duplicate`, `unmatched`.

Definition of done:

- можно спросить: по email/phone кто кандидат;
- конфликт не склеивается автоматически;
- есть отчет по coverage.

### Фаза 4. Mango Call Timeline Adapter

Сделать:

- адаптер из product DB calls;
- связка по телефону/AMO refs;
- события `mango_call`, `call_transcript`, `call_analysis`;
- ссылки на audio/transcript/analysis без копирования тяжелых файлов.

Definition of done:

- можно построить timeline по клиентам из звонков;
- не запускаются ASR/R+A;
- runtime DB не меняется.

### Фаза 5. Mail Timeline Adapter

Сделать после первых артефактов mail-диалога:

- адаптер из mail archive DB;
- события `email_message`, `email_attachment`;
- связка email -> customer identity;
- ambiguous/unmatched report.

Definition of done:

- письма появляются в одной ленте рядом со звонками;
- вложения видны как artifacts;
- raw `.eml` не кладется в git и не дублируется в SQLite.

### Фаза 6. AMO/Tallanto Context Adapter

Сделать:

- события по AMO deal/contact snapshots;
- Tallanto student/payment/group/abonement context;
- opportunity/thread grouping.

Definition of done:

- видно, является ли клиент учеником;
- видно, есть ли активная оплата/абонемент/группа;
- sales context не путается с service context.

### Фаза 7. Timeline Facade and Reports

Сделать:

- `build_customer_timeline_readiness_report`;
- `get_customer_timeline(customer_id)`;
- `search_timeline(q)`;
- summary по клиенту;
- top identity conflicts.

Definition of done:

- CLI строит отчет без UI;
- API facade read-only;
- тесты на filters/search/pagination.

### Фаза 8. Dashboard Panel

Сделать:

- панель Customer Timeline;
- список клиентов;
- карточка клиента;
- timeline events;
- conflict panel;
- next action panel;
- bot context preview.

Definition of done:

- менеджер/РОП может открыть клиента и понять историю без чтения JSON.

### Фаза 9. Bot Context Builder

Сделать:

- compact context для бота;
- recent events;
- active opportunity;
- facts with source refs;
- unresolved promises;
- questions already answered;
- escalation flags.

Definition of done:

- бот получает не весь архив, а компактную проверяемую память;
- каждый факт имеет source event;
- есть offline preview, без live send.

## Первый безопасный work package

После утверждения плана лучше начать с Фазы 1:

```text
src/mango_mvp/customer_timeline/
  __init__.py
  contracts.py
  ids.py
  safety.py

tests/test_customer_timeline_contracts.py
```

Без DB, без чтения реальных писем, без runtime DB, без ASR/R+A и без CRM write.

Почему так:

- это не конфликтует с mail-диалогом;
- не зависит от качества обработки звонков;
- сразу задает общий язык для всех источников;
- последующие адаптеры смогут использовать одни contracts.

## Риски

### Identity conflicts

Один телефон/email может относиться к нескольким ученикам или сделкам.
Решение: не склеивать автоматически, а явно показывать `ambiguous`.

### Timeline overload

Если показать все письма и звонки без группировки, менеджер утонет.
Решение: summary, event importance, filters, grouped threads.

### Bot hallucination risk

Бот не должен использовать неподтвержденные факты как истину.
Решение: bot context только из verified/strong или с явными confidence flags.

### Source drift

AMO/Tallanto/почта меняются со временем.
Решение: import runs, source timestamps, readback/snapshot diffs.

### SQLite growth

SQLite должен хранить метаданные и индексы, не файлы.
Решение: raw files on disk + sha256 + paths + backup/restore commands.

## Итоговая позиция

Unified Customer Timeline должен стать центральным продуктовым слоем Mango
Analyse. Это не новая обработка звонков и не новый почтовый парсер, а read model,
которая собирает результаты всех потоков в одну клиентскую историю.

Сначала строим contracts и store, потом подключаем Mango calls, затем mail,
затем AMO/Tallanto context, затем dashboard и bot context.
