# Channel Bot Strategic Plan

Дата: 2026-05-09

## 1. Executive summary

Legacy `TG_sale_bot` закрывается как отдельный продуктовый путь. В Mango Analyse / AI Office нужно забрать не старую RAG-систему и не Telegram-only runtime, а полезные паттерны:

- Telegram/Telegram Business channel adapter;
- webhook queue, retry и idempotency;
- draft -> approve -> send lifecycle;
- admin inbox;
- Render/Docker/smoke/deploy опыт;
- UX-паттерны клиентского бота и manager approval workspace.

Целевая система Mango строится как Revenue OS / AI Office backend:

```text
Channel Adapter -> Common Request Backend -> Signal Engine -> CRM / Knowledge / Agent Actions
```

Telegram, сайт, CRM-чат, WhatsApp, MAX и другие каналы должны быть адаптерами. Ядро не должно импортировать Telegram SDK, старую RAG-логику или hardcoded сценарии из legacy bot.

## 2. Current Mango state

### 2.1 Call processing contour

По состоянию на текущие артефакты processing-диалога:

- источник в окне 2025-01-01 - 2026-05-31: 64 867 аудио;
- исключено из ASR: 35;
- actionable audio: 64 832;
- ASR done: 64 832;
- Full R+A: 64 832.

Главная текущая проблема уже не массовый ASR, а quality guardrails:

- автоответчики, голосовая почта, недозвоны и ASR-артефакты иногда проходили как `sales_call`, `service_call` или `technical_call`;
- такие строки загрязняли readiness, sales moments, Knowledge Base, ROP pack и потенциальные bot seeds;
- processing-диалог добавил `src/mango_mvp/quality/non_conversation.py`, transcript-quality review pipeline, dry-run/backfill tools и tests.

Важные текущие цифры из аудита:

- terminal analyzed calls: 64 832;
- прежний contentful: 51 429;
- прежний non-conversation: 13 403;
- подозрительных contentful с no-live markers: 3 416;
- hard-gate dry-run на всем корпусе показал 5 463 rows, которые могли бы перейти в `non_conversation`;
- protected live dialogues: 42 855;
- pilot LLM quality pipeline на 1000 disputed calls был намеренно консервативным: 153 auto-apply candidates, 760 Claude/audit-required.

Вывод для bot/channel слоя: будущий бот не должен брать ответы напрямую из старых transcript/RAG артефактов. Он должен использовать только validated/safe signals после quality gates.

### 2.2 SaaS/productization contour

Уже есть:

- Mango shadow poll/capture foundation;
- product appliance SQLite DB;
- read-only Product API и dashboard contracts;
- scheduler/appliance loop foundation;
- AMO доступ через `https://api.fotonai.online` и server-side OAuth refresh;
- Tallanto read-only доступ;
- AMO/Tallanto mapping и writeback guards;
- sales insight / knowledge layer;
- opt-in agent runtime preview: `AgentRun`, `AgentActionPolicy`, `AgentAction`;
- safety matrix, runbooks и response plans.

Чего нет:

- универсального channel runtime;
- `ChannelMessage`, `ChannelSession`, `BotReply`, `ChannelAdapter`;
- безопасного draft approval workspace для сообщений клиентам;
- Telegram adapter без live-send;
- channel-specific webhook queue в Mango.

## 3. What we should not reuse from legacy Telegram bot

Не переносим:

- старую RAG-систему;
- OpenAI vector store metadata;
- старую SQLite DB;
- `.env`, `data/`, secrets;
- Telegram-only `bot.py` как core;
- прямую отправку сообщений без approval;
- Telegram Mini App как единственный admin interface;
- hardcoded УНПК/МФТИ assumptions как platform core.

Причина: Mango строится не как FAQ/RAG bot, а как система обработки пользовательских вопросов и сигналов, связанная с CRM, звонками, результатами продаж, политиками безопасности и действиями менеджеров.

## 4. What we should reuse

Используем как reference:

- webhook vs polling deployment lessons;
- Render Docker/startup/smoke patterns;
- Telegram Business `business_connection_id`/thread mapping;
- draft -> approve -> send;
- idempotency;
- admin inbox UX;
- follow-up queue UX;
- runtime diagnostics;
- release smoke checks;
- graceful degradation/fallback patterns;
- Lead Radar idea, но переписанную поверх Mango signals.

Если legacy source code будет передан отдельно, его надо положить не в `src/`, а в read-only handoff/source папку и переносить только малые проверяемые фрагменты.

## 5. Target signal architecture

Mango должен обрабатывать не только вопрос пользователя, а весь контекст сигнала:

```text
Inbound message / call / CRM event
 -> normalized event
 -> identity resolution
 -> context assembly
 -> signal extraction
 -> policy and safety gates
 -> draft reply / recommended action
 -> manager approval
 -> controlled execution
 -> outcome feedback
```

Типы сигналов:

- явный вопрос клиента;
- тема/интент;
- стадия сделки;
- срочность;
- риск потери выручки;
- no-live/non-conversation;
- follow-up due;
- повторное возражение;
- запрос цены/скидки/рассрочки/возврата;
- brand/legal/personal-data risk;
- связь с Mango звонками;
- связь с AMO/Tallanto.

Ответ бота должен собираться не из старой RAG-базы, а из:

- validated knowledge из звонков;
- owner/ROP-approved answer base;
- актуального CRM/Tallanto context;
- tenant policy;
- safety restrictions;
- manager approval state.

## 6. Hosting decision

### Recommendation

Production bot/channel backend лучше размещать на нашем контуре: `api.fotonai.online` / VPS / будущий AI Office hosting.

Render оставить как:

- reference deployment pattern;
- временный sandbox для legacy bot, если нужно сравнить поведение;
- источник smoke/deploy практик.

### Why not Render as final home

Render удобен для быстрого MVP, но для Mango/AI Office слабее как целевой production контур:

- AMO OAuth уже живет на `api.fotonai.online`;
- Tallanto/Mango/API keys должны управляться централизованно;
- нужны фоновые jobs, audit logs, scheduler и product DB;
- потребуется единая observability и backup policy;
- клиент-hosted/appliance модель должна быть управляемой из одного архитектурного подхода;
- Telegram adapter не должен жить отдельно от CRM/action runtime.

### Target deployment shape

Вариант для первого production контура:

```text
https://api.fotonai.online/api/channels/telegram/webhook
https://api.fotonai.online/api/channels/telegram/drafts
https://api.fotonai.online/api/agent/actions/preview
https://api.fotonai.online/api/integrations/amocrm/status
```

Если нужен отдельный домен для бота:

```text
https://bot.fotonai.online/telegram/webhook
```

Но backend все равно должен использовать общий AI Office runtime, а не отдельную старую RAG/SQLite систему.

## 7. Repository organization

Handoff-пакет legacy bot должен храниться как внешний reference:

```text
_external_handoffs/telegram_bot_legacy_20260509/
```

Если позже будет передан source code legacy bot:

```text
_external_handoffs/telegram_bot_legacy_20260509/source_readonly/
```

Правило: код из `source_readonly` нельзя импортировать напрямую из Mango. Перенос только через новый код, tests и review.

## 8. Development phases

### Phase 1. Channel contracts

Создать новый изолированный модуль:

```text
src/mango_mvp/channels/
```

Контракты:

- `ChannelMessage`;
- `ChannelAttachment`;
- `ChannelSession`;
- `BotReply`;
- `ReplyButton`;
- `RecommendedAction`;
- `ChannelAdapter`.

Не создавать второй `AgentAction`, потому что в `amocrm_runtime` уже есть SQLAlchemy model `AgentAction`. Для channel layer использовать `RecommendedAction`, затем маппить его в `ActionProposal`.

Definition of done:

- contracts;
- fake adapter;
- unit tests;
- no network;
- no live send.

### Phase 2. Common bot preview backend

Сделать сервис:

```text
ChannelMessage -> BotPreview
```

Первый preview должен:

- определить thread/session;
- собрать минимальный context;
- вернуть draft reply;
- вернуть recommended actions;
- пометить `requires_approval=true`;
- не отправлять сообщение.

Источники context на старте:

- AMO status/context через AI Office API;
- Tallanto read-only;
- product DB;
- validated/safe knowledge artifacts.

### Phase 3. Telegram read-only adapter

Сделать только parser:

- обычный Telegram message update;
- Telegram Business message update;
- Mini App `web_app_data` как structured event;
- duplicate/idempotency key;
- invalid payload handling.

Запрещено:

- отправлять сообщения;
- использовать production bot token;
- включать polling/webhook live без отдельного approval.

### Phase 4. Draft approval workspace

Нужно связать channel draft с существующим agent runtime:

- draft created;
- needs_review;
- approved;
- rejected;
- sent/mock_sent;
- failed.

Сначала хранить как product/channel tables или JSON report, затем решать, где постоянная БД.

Обязательно:

- audit trail;
- idempotency;
- no-live-send default;
- manager visible context.

### Phase 5. Controlled Telegram send

Только после Phase 1-4:

- test bot или test Business account;
- explicit approval;
- send idempotency key;
- rollback/disable flag;
- limited allowlist;
- smoke test.

### Phase 6. Site/CRM adapters

Добавить второй канал не Telegram:

- site chat или CRM chat.

Цель: доказать, что core не Telegram-only.

### Phase 7. Signal engine v1

Ввести канонический слой сигналов:

- `CustomerSignal`;
- `SignalEvidence`;
- `SignalPolicy`;
- `SignalDecision`;
- `SafeAnswer`.

Этот слой должен использовать call quality v2, CRM context и channel messages.

### Phase 8. Revenue feedback loop

Связать ответы/действия с outcome:

- клиент ответил/не ответил;
- менеджер approved/rejected/edited;
- задача создана/закрыта;
- сделка продвинулась/закрылась;
- ROP отметил ответ как хороший/опасный.

## 9. Dependency on processing quality

До завершения processing-quality acceptance bot может работать только в conservative mode:

- no autonomous commercial promises;
- no prices/discounts/refunds/deadlines from unvalidated examples;
- no use of raw `ideal_answer_example` as bot reply;
- no no-live calls in bot seeds;
- all client-facing replies require approval.

После processing acceptance:

- rebuild readiness v2;
- rebuild sales moments v2;
- rebuild Knowledge Base v3;
- rebuild ROP pack v2;
- produce bot-safe answer base;
- enable channel preview against validated knowledge.

## 10. Immediate next work package

Recommended next step for this dialog:

1. Keep legacy handoff under `_external_handoffs/telegram_bot_legacy_20260509/`.
2. Add `src/mango_mvp/channels/` contracts.
3. Add fake adapter and preview service.
4. Add tests:
   - contract validation;
   - fake inbound -> preview -> draft;
   - idempotency;
   - no-live-send guarantee.
5. Add docs for channel architecture.

This can be done while processing-dialog continues improving transcript quality, because it does not touch ASR/R+A, runtime DB, CRM writes or live Telegram.

## 11. Strategic decision

TG sale bot as a separate product should be closed. Mango/AI Office becomes the main system.

Legacy bot contributes:

- deployment experience;
- Telegram adapter patterns;
- UI/UX ideas;
- approval lifecycle patterns.

Mango owns:

- CRM/Tallanto/AMO context;
- Mango call intelligence;
- quality gates;
- signal engine;
- action policies;
- channel adapters;
- product dashboard;
- future SaaS/client-hosted packaging.
