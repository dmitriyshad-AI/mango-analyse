# Как отделить Telegram-бота от общего ядра

## 1. Проблема

Сейчас Telegram - первый живой канал, поэтому есть риск, что логика ИИ-сотрудника начнёт срастаться с Telegram-обвязкой.

Этого нельзя допустить. Telegram должен быть только одним способом доставки сообщений. Бизнес-логика должна жить в общем ядре.

## 2. Целевое разделение

```text
Telegram / сайт / email / CRM chat
-> ChannelAdapter
-> ConversationSession
-> ContextBuilder
-> KnowledgeAccess
-> PolicyEngine
-> AnswerEngine
-> PostFilter
-> HumanHandoff
-> FeedbackStore
-> AuditLog
```

## 3. Слои и ответственность

### 3.1. ChannelAdapter

Ответственность:

- принять событие внешнего канала;
- нормализовать в `ChannelMessage`;
- отрендерить ответ в формат канала;
- не принимать бизнес-решений.

Текущие реализации:

- Telegram Bot API;
- Telegram Business parser;
- Telegram history parser;
- Web/CRM chat adapter.

Запрещено:

- выбирать route;
- решать, можно ли автономно ответить;
- читать базу знаний напрямую;
- писать в AMO/Tallanto;
- отправлять live-сообщения в обход ядра.

### 3.2. ConversationSession

Ответственность:

- связать сообщения в диалог;
- хранить recent messages;
- хранить бренд;
- хранить известные поля клиента;
- хранить режим пилота/теста;
- обеспечивать idempotency.

Текущий Telegram-пилот хранит историю в памяти процесса. Следующий шаг - постоянное локальное хранилище для сессий.

### 3.3. ContextBuilder

Ответственность:

- собрать единый context pack;
- не раскрывать внутренние данные клиенту;
- отметить надёжность источников;
- не делать live-write.

Источники:

- текущий канал;
- recent messages;
- база знаний;
- gold-ответы;
- AMO read-only;
- Tallanto read-only;
- local runtime summary;
- customer timeline;
- правила РОПа;
- матрица автономности.

### 3.4. KnowledgeAccess

Ответственность:

- дать только client-safe факты активного бренда;
- вернуть версию базы знаний;
- вернуть gold-правила как стиль, а не как дословный шаблон;
- не смешивать Фотон и УНПК;
- возвращать missing facts.

Минимальный контракт:

```text
get_context(
  brand_id,
  topic_id,
  question_text,
  knowledge_release_id
) -> facts, gold_rules, missing_facts, safety
```

### 3.5. PolicyEngine

Ответственность:

- применить матрицу автономности;
- определить route;
- запретить P0;
- запретить mixed-brand;
- запретить ответ без факта;
- отличить новый лид от известного клиента.

PolicyEngine должен быть единственным местом, где решается:

- `bot_answer_self_for_pilot`;
- `draft_for_manager`;
- `manager_only`;
- `blocked`.

### 3.6. AnswerEngine

Ответственность:

- подготовить ответ по контексту;
- использовать gold-правила как ориентир качества;
- отвечать живо, коротко и по делу;
- не выдумывать факты;
- не спрашивать повторно известное.

AnswerEngine не имеет права сам повышать автономность.

### 3.7. PostFilter

Ответственность:

- проверить уже сгенерированный текст;
- заблокировать опасное;
- добавить flags;
- при необходимости понизить route.

PostFilter должен быть общим для всех каналов.

### 3.8. HumanHandoff

Ответственность:

- сформировать понятную карточку менеджеру;
- объяснить, почему вопрос передан;
- показать недостающие факты;
- не раскрывать клиенту служебные поля.

Для Telegram сейчас это может быть служебное сообщение или report. Для будущего рабочего места - карточка в UI.

### 3.9. FeedbackStore

Ответственность:

- сохранить оценку сотрудника;
- сохранить исправленный ответ;
- выделить ошибки;
- отправить ошибку в очередь тестов/КБ.

## 4. Карта текущих модулей

| Целевой слой | Уже есть в проекте |
|---|---|
| ChannelAdapter | `telegram_adapter.py`, `web_chat_adapter.py`, `telegram_bot_polling.py` |
| ConversationSession | `contracts.py`, `persistence.py`, memory/session logic |
| ContextBuilder | `pilot_context.py`, `telegram_pilot_context_builder.py`, `customer_context_for_draft.py` |
| KnowledgeAccess | `knowledge_base`, `fact_registry`, KB snapshot/bot pack |
| PolicyEngine | autonomy matrix, `draft_prompt_builder.py`, guards in `subscription_llm.py` |
| AnswerEngine | `subscription_llm.py` |
| PostFilter | `subscription_llm.py`, safety tests, semantic rules |
| HumanHandoff | `telegram_manager_inbox.py`, reports |
| FeedbackStore | feedback/report modules, pilot feedback TZ |
| AuditLog | audit packs, local logs |

## 5. Что нужно выделить следующим инженерным этапом

Без переписывания всего проекта нужно постепенно ввести общий сервис:

```text
AiEmployeeCore.handle_inbound_message(context) -> DecisionEnvelope
```

Где `DecisionEnvelope` содержит:

- route;
- answer_text;
- manager_card;
- safety_flags;
- semantic_flags;
- used_facts;
- missing_facts;
- required_human_action;
- audit_refs;
- latency;
- store_refs.

Telegram должен вызывать этот сервис, а не собирать решение внутри себя.

## 6. Правило для будущих каналов

Новый канал считается правильным, если он:

1. Использует общий `ChannelMessage`.
2. Не имеет своих бизнес-правил автономности.
3. Не читает базу знаний напрямую.
4. Не отправляет live-сообщение без разрешения ядра.
5. Пишет все решения в общий журнал.
6. Проходит те же P0/brand/semantic gates, что Telegram.

## 7. Что нельзя делать

Нельзя:

- делать отдельного “email-бота” со своими правилами;
- делать отдельного “site-chat бота” со своими промптами;
- размножать матрицу автономности;
- хранить разные версии базы знаний по каналам без release id;
- разрешать live-send на уровне адаптера;
- использовать Telegram-specific поля как основной доменный формат.

## 8. Минимальный безопасный шаг

Первый шаг отделения ядра:

1. Описать `DecisionEnvelope`.
2. Зафиксировать вход `PilotContext`.
3. Сделать один фасад `build_ai_employee_decision`.
4. Подключить Telegram к фасаду без изменения поведения.
5. Покрыть тестом: Telegram и fake web-chat дают одинаковый route на одинаковом контексте.

Это можно делать после стабилизации текущего пилота, не раньше.
