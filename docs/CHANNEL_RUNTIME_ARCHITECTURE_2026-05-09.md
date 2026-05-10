# Channel Runtime Architecture

Дата: 2026-05-09

## Назначение

Этот документ фиксирует первые восемь шагов channel/bot направления. Mango Analyse / AI Office строит не отдельный Telegram-бот, а общий runtime для сообщений из разных каналов:

```text
Telegram / сайт / CRM-чат / WhatsApp / MAX
 -> Channel Adapter
 -> Common Channel Contracts
 -> Bot Preview / Approval / Agent Actions
```

Этапы 1-8 реализуют common contracts, безопасный draft-preview, слой рекомендованных действий, draft approval workspace, read-only Telegram adapter, второй read-only канал site/CRM chat, Signal Engine v1 и Revenue feedback loop. Live Telegram, реальная отправка сообщений, RAG и CRM-запись в эти этапы не входят.

## Что добавлено в этапе 1

Код:

- `src/mango_mvp/channels/contracts.py`
- `src/mango_mvp/channels/__init__.py`

Тесты:

- `tests/test_channels_contracts.py`

Основные контракты:

- `ChannelMessage` - единый формат входящего или исходящего сообщения.
- `ChannelSession` - единый формат диалога/треда в канале.
- `BotReply` - канало-независимый черновик ответа.
- `ReplyButton` - кнопка без привязки к Telegram.
- `RecommendedAction` - рекомендация действия, не SQLAlchemy model и не `AgentAction`.
- `ChannelAdapter` - интерфейс будущего адаптера канала.

Поддерживающие контракты:

- `ChannelAttachment`
- `ChannelRenderedReply`
- `SendResult`

Поддерживающие функции:

- `stable_message_idempotency_key` - стабильный ключ для защиты от дублей.
- `dedupe_channel_messages` - удаляет повторные сообщения, сохраняя первый экземпляр.
- `ChannelSession.from_message` - создает сессию из нормализованного сообщения.

## Что добавлено в этапе 2

Код:

- `src/mango_mvp/channels/preview_service.py`

Тесты:

- `tests/test_channels_preview_service.py`

Основные сущности:

- `ChannelDraftPreview` - безопасный черновик ответа для проверки менеджером.
- `ChannelPreviewService` - сервис, который принимает `ChannelMessage` и создает `BotReply`.
- `build_channel_draft_preview` - helper для простого вызова preview.
- `stable_draft_id` - стабильный draft id для защиты от дублей.

Поведение этапа 2:

- принимает только inbound messages;
- создает deterministic draft;
- добавляет `RecommendedAction(action_type="draft_client_message")`;
- всегда ставит `requires_approval=true`;
- исключает `raw_payload` из JSON по умолчанию;
- не использует LLM/RAG;
- не делает network calls;
- не отправляет сообщение клиенту.

## Что добавлено в этапе 3

Код:

- `src/mango_mvp/channels/actions.py`

Тесты:

- `tests/test_channels_actions.py`

Типы рекомендованных действий:

- `draft_client_message` - подготовить черновик ответа.
- `request_crm_context` - запросить read-only CRM/Tallanto context.
- `handoff_to_manager` - передать тред менеджеру.
- `create_follow_up_task` - подготовить follow-up задачу.
- `mark_manual_review` - отправить на ручную проверку из-за коммерческого/safety-сигнала.
- `notify_rop_hot_lead` - подсветить РОПу горячий сигнал.

Каждое действие имеет channel-level policy:

- autonomy level;
- requires approval;
- requires notification;
- `live_execution_allowed=false`.

Добавлен mapping:

```text
RecommendedAction -> ActionProposal
```

Это только преобразование данных. Оно не создает `AgentRun`, не пишет в БД и не выполняет действие.

## Что добавлено в этапе 4

Код:

- `src/mango_mvp/channels/storage.py`

Тесты:

- `tests/test_channels_storage.py`

Основные сущности:

- `ChannelMemoryStore` - reference-хранилище для будущего inbox/approval workspace.
- `ChannelMessageRecord` - сохраненное normalized сообщение.
- `ChannelDraftRecord` - черновик ответа со статусом lifecycle.
- `ChannelActionRecord` - сохраненная recommended action со статусом review/execution-preview.
- `ChannelHistoryEvent` - audit trail по сообщениям, черновикам и действиям.
- `ChannelPreviewStoreResult` - результат idempotent записи preview.

Lifecycle черновика:

```text
needs_review -> approved -> mock_sent
needs_review -> rejected
needs_review -> failed
needs_review -> superseded
approved -> failed/superseded
```

Lifecycle recommended action:

```text
proposed -> approved -> mock_executed
proposed -> rejected/blocked/dismissed/failed
approved -> failed
```

Реальные статусы `sent`, `live_sent`, `executed` в этом слое запрещены. Для демонстрации workspace есть только `mock_sent` и `mock_executed`.

Что хранится для менеджера:

- канал и thread;
- id пользователя в канале;
- исходный текст обращения;
- черновик ответа;
- список recommended actions;
- safety flags;
- blocked reasons;
- manager review context;
- history событий.

`ChannelMemoryStore.snapshot()` отдает JSON-ready состояние для будущего UI/API. `raw_payload` исключается по умолчанию.

## Что добавлено в этапе 5

Код:

- `src/mango_mvp/channels/telegram_adapter.py`

Тесты:

- `tests/test_channels_telegram_adapter.py`

Основная сущность:

- `TelegramReadOnlyAdapter` - parser/renderer skeleton без Telegram SDK, токенов, polling, webhook и live send.

Что умеет parser:

- обычный Telegram `message` -> `ChannelMessage(channel="telegram_bot")`;
- `edited_message` -> отдельный read-only event;
- Telegram Business `business_message` -> `ChannelMessage(channel="telegram_business")`;
- Telegram Business `edited_business_message` -> отдельный read-only event;
- Telegram Mini App `web_app_data` -> `ChannelMessage(channel="telegram_miniapp")` со structured metadata;
- attachments без текста: `photo`, `document`, `voice`, `audio`, `video`, `sticker`;
- service/unsupported updates безопасно пропускаются как empty sequence.

Что умеет renderer:

- превращает `BotReply` в Telegram-like payload для `sendMessage`;
- добавляет `business_connection_id` для business thread;
- превращает `ReplyButton` в inline keyboard payload;
- создает стабильный render idempotency key;
- явно помечает `live_send_enabled=false`.

`send()` в этом adapter всегда возвращает `SendResult(sent=false)`. Даже если вызвать `send(..., live_send_enabled=true)`, он вернет `live_send_not_implemented`, потому что реальная отправка относится к отдельному controlled-send этапу.

## Что добавлено в этапе 6

Код:

- `src/mango_mvp/channels/web_chat_adapter.py`

Тесты:

- `tests/test_channels_web_chat_adapter.py`

Основная сущность:

- `WebChatReadOnlyAdapter` - read-only adapter для `site_chat` и `crm_chat` payload без webhook server, CRM write и live send.

Что умеет parser:

- nested site widget payload -> `ChannelMessage(channel="site_chat")`;
- flat CRM chat payload -> `ChannelMessage(channel="crm_chat")`;
- несколько распространенных имен полей: `message_id`, `id`, `thread_id`, `conversation_id`, `chat_id`, `lead_id`, `visitor_id`, `contact_id`, `body`, `content`;
- timestamps в ISO/Z и unix timestamp;
- attachments без текста через `attachments`/`files`;
- service events `typing/read/delivered/presence/heartbeat` безопасно пропускаются.

Что умеет renderer:

- превращает `BotReply` в generic `draft_reply` payload;
- сохраняет buttons как channel-neutral actions;
- создает стабильный render idempotency key;
- явно помечает `live_send_enabled=false`.

`send()` в web/CRM adapter всегда возвращает `SendResult(sent=false)`. Даже если вызвать `send(..., live_send_enabled=true)`, он вернет `live_send_not_implemented`.

Этап 6 доказывает, что ядро `ChannelMessage -> preview -> storage -> render` не зависит от Telegram.

## Что добавлено в этапе 7

Код:

- `src/mango_mvp/channels/signals.py`

Тесты:

- `tests/test_channels_signals.py`

Основные сущности:

- `SignalEvidence` - доказательство сигнала: source, excerpt, markers, weight.
- `SignalPolicy` - safe policy для каждого типа сигнала.
- `CustomerSignal` - канонический сигнал клиента.
- `SafeAnswer` - только draft-answer, всегда с `requires_approval=true`.
- `SignalDecision` - итоговое решение по сообщению: сигналы, safe answer, policy flags, blocked reasons.

Типы сигналов:

- `need_crm_context`;
- `customer_question`;
- `urgency`;
- `commercial_risk`;
- `manager_handoff`;
- `follow_up`;
- `hot_lead`;
- `attachment_received`.

Источники сигналов:

- текст `ChannelMessage`;
- вложения;
- `RecommendedAction`;
- read-only context.

Что принципиально важно:

- `draft_client_message` сам по себе не считается вопросом клиента;
- вопрос клиента определяется по текстовым маркерам;
- коммерческие, handoff, follow-up и hot-lead сигналы могут усиливаться recommended actions;
- context-only signals работают без preview/actions;
- `priority`/`lead_priority` превращается в `hot_lead` только для `hot`, `high`, `p0`, `горячий`, `срочно`;
- `SafeAnswer` запрещает non-draft и ответы без approval;
- policy запрещает autonomous reply и live execution.

Signal Engine v1 не пишет в CRM/Tallanto, не делает network calls, не использует LLM/RAG и не отправляет ответы клиенту.

## Что добавлено в этапе 8

Код:

- `src/mango_mvp/channels/feedback.py`

Тесты:

- `tests/test_channels_feedback.py`

Основные сущности:

- `FeedbackEvent` - единый факт обратной связи по сообщению, черновику, действию, решению или сделке.
- `ChannelFeedbackMemoryStore` - in-memory журнал feedback events с idempotency.
- `ChannelFeedbackReport` - отчет по feedback loop для будущего UI/API.
- `FeedbackStoreResult` - результат безопасной записи события.

Типы feedback events:

- `manager_draft_approved` - менеджер одобрил черновик.
- `manager_draft_rejected` - менеджер отклонил черновик.
- `manager_draft_edited` - менеджер отредактировал черновик.
- `action_accepted` - recommended action приняли в работу.
- `action_dismissed` - recommended action отклонили.
- `client_replied` - клиент ответил после контакта.
- `client_no_reply` - клиент не ответил.
- `follow_up_done` - follow-up выполнен.
- `follow_up_overdue` - follow-up просрочен.
- `lead_moved` - сделка сдвинулась по этапу, только read-only import.
- `lead_won` - сделка выиграна, только read-only import.
- `lead_lost` - сделка проиграна, только read-only import.
- `rop_answer_good` - РОП отметил ответ как хороший.
- `rop_answer_risky` - РОП отметил ответ как рискованный.

Поддерживающие функции:

- `build_manager_draft_feedback_event` - feedback по draft-review.
- `build_action_feedback_event` - feedback по recommended action.
- `build_decision_feedback_event` - feedback по `SignalDecision`.
- `build_read_only_lead_outcome_event` - read-only/imported outcome из CRM.
- `summarize_feedback_events` - сводные метрики.
- `build_feedback_loop_report` - отчет для будущей панели продукта.
- `feedback_loop_safety_contract` - safety contract этапа.

Что считает summary:

- сколько черновиков одобрили/отклонили/отредактировали;
- acceptance rate по recommended actions;
- reply/no-reply rate клиента;
- done/overdue rate по follow-up;
- moved/won/lost outcome по сделкам;
- good/risky оценки РОПа;
- positive/risk feedback events.

Что принципиально важно:

- lead outcomes разрешены только как `imported_read_only=true`;
- feedback metadata не может заявлять live-send, CRM write, Tallanto write, runtime DB write, ASR/R+A или network side effect;
- raw provider payload запрещен в feedback metadata;
- store остается process-local и не пишет в файлы/БД;
- это слой обучения продукта, а не слой исполнения.

Revenue feedback loop не пишет в CRM/Tallanto, не делает network calls, не использует LLM/RAG, не запускает ASR/R+A и не отправляет ответы клиенту.

## Важное архитектурное решение

В Mango уже есть `AgentAction` в `src/mango_mvp/amocrm_runtime/agent_models.py`. Поэтому channel layer использует имя `RecommendedAction`.

Будущий mapping:

```text
RecommendedAction -> ActionProposal -> AgentAction
```

Так мы не создаем две разные сущности `AgentAction` с разным смыслом.

## Safety

Этапы 1-8:

- не импортирует Telegram SDK;
- не делает network calls;
- не отправляет сообщения;
- не пишет в CRM;
- не пишет в Tallanto;
- не запускает ASR/R+A;
- не пишет в runtime DB;
- не трогает `stable_runtime`.

`SendResult` по умолчанию описывает неуспешную/заблокированную отправку. Реальная отправка появится только в будущих этапах после approval workflow.

`ChannelDraftPreview` всегда создается в статусе `needs_review`, а `live_send` в safety-contract остается `false`.

`ChannelActionPolicy` также запрещает `live_execution_allowed=true`, чтобы action layer не стал скрытым обходом approval workflow.

`ChannelMemoryStore` не пишет в файлы и БД. Это reference storage для проверки lifecycle и будущего UI. Persistent SQLite/product storage нужно подключать отдельным этапом после утверждения схемы таблиц.

`ChannelFeedbackMemoryStore` тоже не пишет в файлы и БД. Он нужен, чтобы проверить revenue feedback semantics до подключения persistent product storage.

## Почему не переносим старый RAG

Старый Telegram-проект используется как reference для UX/deploy/adapter patterns. Его RAG-система не переносится, потому что Mango должен отвечать на основе:

- validated knowledge из звонков;
- quality gates;
- AMO/Tallanto context;
- owner/ROP-approved answer base;
- safety policy.

До завершения processing-quality работ любые client-facing ответы должны оставаться черновиками с подтверждением человека.

## Что делать после этапа 8

План из 8 этапов channel foundation закрыт. Следующий шаг уже не новый channel phase, а закрепление foundation в продуктовый контур:

1. Persistent product storage для channel messages, drafts, actions, signals и feedback events в отдельной product DB, не в runtime DB.
2. Read-only HTTP API для inbox/approval/feedback dashboard.
3. UI approval workspace: список обращений, черновик, сигналы, recommended actions, feedback history.
4. Интеграция с AI Office API для read-only CRM context через `api.fotonai.online`, без прямого AMO token и без CRM write.
5. После acceptance обработки звонков - подключить validated knowledge base к preview, сохранив manager approval и safety gates.
