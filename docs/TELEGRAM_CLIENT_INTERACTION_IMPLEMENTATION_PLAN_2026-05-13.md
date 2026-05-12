# Telegram Client Interaction Implementation Plan (план реализации взаимодействия с клиентами в Telegram)

Дата: 2026-05-13
Контур: Mango Analyse / AI Office (офис ИИ)
Статус: техническое задание для реализации MVP (минимально жизнеспособного продукта)

Примечание: точные имена API-полей, методов, классов, переменных окружения и
будущих модулей оставлены в оригинальном написании, потому что это внешние
контракты или имена кода. Русский смысл указан рядом в скобках.

## Решение После Уточнения

MVP (минимально жизнеспособный продукт) должен включать сразу два контура:

1. Telegram Business connected bot (подключенный бизнес-бот Telegram) для
   официального ingest (приема сообщений), business update parsing (парсинга
   бизнес-обновлений), draft pipeline (конвейера черновиков), audit (аудита) и
   будущей controlled send (контролируемой отправки).
2. `TelegramNativeDraftWriter` (сервис записи нативных черновиков Telegram) на
   TDLib (официальной библиотеке Telegram-клиента), который умеет только:
   `save_draft` (сохранить черновик), `clear_draft` (очистить черновик) и
   `get_draft_state` (получить состояние черновика).

Native draft (нативный черновик в поле ввода Telegram) теперь входит в MVP.
Менеджер должен открыть обычный Telegram-чат с клиентом, увидеть подготовленный
текст в поле ввода, проверить его и отправить вручную кнопкой Send (отправить)
в самом Telegram. Сервис не должен иметь команду отправки.

OpenClaw не входит в production path (боевой путь) и не получает доступ к
пользовательской сессии основного Telegram-аккаунта.

## Главная Цель MVP

```text
клиент пишет в основной Telegram-аккаунт по номеру
-> Mango принимает сообщение безопасным официальным способом
-> Mango сопоставляет клиента и собирает контекст
-> Mango создает AI draft (черновик ИИ)
-> Mango сохраняет draft/audit в channel store (хранилище каналов)
-> TelegramNativeDraftWriter кладет текст в native draft (нативный черновик)
-> менеджер открывает обычный Telegram, проверяет текст и вручную отправляет
```

Критерий успеха MVP: менеджер получает черновик прямо в исходном Telegram-чате,
а система при этом не отправляет сообщения автоматически, не пишет в CRM/Tallanto,
не запускает ASR/R+A и не использует legacy RAG (старую RAG-систему).

## Ключевая Архитектурная Граница

Фактический текущий код Mango уже использует `ChannelDraftPreview`
(предпросмотр канального черновика) и `ChannelDraftRecord` (запись канального
черновика). В этом ТЗ короткое имя `ChannelDraft` (канальный черновик) означает
эту пару сущностей, пока не будет введен отдельный новый класс. Именно она
является source of truth (источником правды): в ней хранятся текст, статус,
source refs (ссылки на источники), confidence flags (флаги уверенности),
conflict flags (флаги конфликтов), idempotency key (идемпотентный ключ) и audit
trail (журнал аудита).

Native draft (нативный черновик) в Telegram является только projection
(проекцией) этого `ChannelDraft` в интерфейс менеджера. Telegram draft не
становится источником правды и не содержит бизнес-логики.

Граница компонентов:

- `TelegramBusinessAdapter` (адаптер Telegram Business): принимает и нормализует
  входящие клиентские сообщения.
- `DraftService` (сервис черновиков): строит и хранит `ChannelDraftPreview` /
  `ChannelDraftRecord`; если появится новый фасад `ChannelDraft`, он не должен
  ломать существующие контракты.
- `NativeDraftOrchestrator` (оркестратор нативных черновиков): решает, можно ли
  записать нативный черновик, проверяет idempotency (идемпотентность), конфликты
  и feature flags (флаги функций).
- `TelegramNativeDraftWriter` (сервис записи нативных черновиков): тупой TDLib
  adapter (адаптер TDLib), который выполняет только `save_draft`,
  `clear_draft`, `get_draft_state`.
- `Manager` (менеджер): единственный субъект, который отправляет сообщение в
  обычном Telegram в MVP.

`TelegramNativeDraftWriter` не вызывается напрямую AI (ИИ), RAG, OpenClaw, CRM,
UI или произвольным агентом. Только внутренний `NativeDraftOrchestrator` создает
`TelegramNativeDraftIntent` (намерение нативного черновика).

## Что Уже Готово В Проекте

Не начинать с нуля. В проекте уже есть foundation (фундамент):

- `src/mango_mvp/channels/contracts.py` - `ChannelMessage` (канальное сообщение),
  `ChannelSession` (канальная сессия), `BotReply` (ответ/черновик бота),
  `RecommendedAction` (рекомендованное действие), `ChannelAdapter`
  (адаптер канала), idempotency helpers (помощники идемпотентности).
- `src/mango_mvp/channels/telegram_adapter.py` - read-only adapter
  (адаптер только для чтения), который парсит обычные Telegram updates
  (обновления Telegram), Telegram Business messages (бизнес-сообщения Telegram),
  edited messages (измененные сообщения), Mini App `web_app_data`
  (данные мини-приложения) и attachments (вложения). Он уже умеет построить
  `sendMessage` payload (payload для отправки сообщения) с
  `business_connection_id` (идентификатором бизнес-соединения), но `send()`
  заблокирован.
- `src/mango_mvp/channels/storage.py` - lifecycle (жизненный цикл) сообщений,
  черновиков, рекомендованных действий и history events (событий истории).
  Реальные статусы `sent`, `live_sent`, `executed` запрещены.
- `src/mango_mvp/channels/persistence.py` - `ChannelSQLiteStore`
  (SQLite-хранилище каналов), которое запрещает `stable_runtime`, не сохраняет
  `raw_payload` по умолчанию и не пишет во внешние системы.
- `src/mango_mvp/channels/preview_service.py` - безопасный deterministic draft
  (детерминированный черновик) без LLM/RAG и без сетевых вызовов.
- `src/mango_mvp/channels/actions.py` - recommended actions (рекомендованные
  действия): черновик ответа, запрос контекста, handoff (передача менеджеру),
  follow-up (следующее касание), manual review (ручная проверка), hot lead
  (горячий лид).
- `src/mango_mvp/channels/workspace.py` - read-only workspace summary
  (сводка рабочего места только для чтения).
- `src/mango_mvp/channels/signals.py` - signal engine (движок сигналов) для
  вопросов клиента, коммерческого риска, срочности, handoff (передачи
  менеджеру) и необходимости CRM-контекста без live actions (реальных действий).
- `src/mango_mvp/channels/feedback.py` - read-only feedback loop (контур
  обратной связи) для событий менеджера и результата лида без внешних записей.
- `src/mango_mvp/channels/telegram_history.py` - historical Telegram import
  (импорт исторического Telegram), inventory (инвентаризация), identity matching
  (сопоставление личности), timeline-ready events (события для ленты клиента).
- `src/mango_mvp/customer_timeline/contracts.py`, `store.py`, `ingestion.py`,
  `read_api.py` - Unified Customer Timeline (единая лента клиента), где уже есть
  типы `TELEGRAM_MESSAGE` (сообщение Telegram), `TELEGRAM_DIALOG` (диалог
  Telegram), `BOT_DRAFT` (черновик бота), `IdentityLinkType.TELEGRAM_USER_ID`
  (связь по ID пользователя Telegram) и `IdentityLinkType.TELEGRAM_USERNAME`
  (связь по имени пользователя Telegram).
- `src/mango_mvp/customer_timeline/approved_context_pack.py` и
  `approval_workspace.py` - approved context pack (утвержденный пакет контекста)
  и рабочее место проверки контекста перед использованием в канальном черновике.
- `src/mango_mvp/customer_timeline/channel_preview_from_pack.py` - bridge
  (мост) из approved Customer Timeline context pack (утвержденного пакета
  контекста клиента) в channel draft (канальный черновик).

Проверенный текущий focused test set (набор фокусных тестов): 55 tests passed
(55 тестов прошли) по Telegram/channel/draft слоям.

## Сверка С Уже Реализованными Скриптами И Модулями

ТЗ согласовано не только с `src/mango_mvp/channels/`, но и с уже написанными
скриптами соседних диалогов. Правила использования:

- `scripts/customer_timeline_import.py`, `customer_timeline_read_report.py`,
  `customer_timeline_approved_context_pack.py`,
  `customer_timeline_approval_workspace.py`,
  `customer_timeline_channel_preview_from_pack.py`,
  `customer_timeline_preview_quality_audit.py` - можно использовать как
  read-only tooling (инструменты только для чтения) вокруг Unified Customer
  Timeline (единой ленты клиента) и draft preview (предпросмотра черновика).
- `scripts/build_channel_workspace_demo.py` - можно использовать как локальный
  demo/smoke (демо/smoke-проверку) рабочего места без live send (реальной
  отправки) и без CRM writes (записей в CRM).
- `scripts/build_telegram_outreach_pack.py`,
  `scripts/build_telegram_openclaw_final.py`,
  `scripts/build_telegram_high_utility_drafts.py`,
  `scripts/merge_telegram_live_enrichment_chunks.py`,
  `scripts/enrich_telegram_phones_live.py` - это legacy/enrichment tooling
  (старые инструменты обогащения) и reference artifacts (справочные артефакты),
  а не часть нового runtime (боевого контура).
- `scripts/enrich_telegram_phones_live.py` и
  `scripts/build_telegram_openclaw_final.py` умеют обращаться к AMO/Tallanto и
  писать отчеты в `stable_runtime/audits`; в рамках MVP их нельзя запускать без
  отдельного явного разрешения владельца.
- `_external_handoffs/telegram_bot_legacy_20260509/` используется только как
  handoff/reference (пакет передачи знаний), без импорта legacy source
  (старого исходного кода) в Mango core (ядро Mango).
- `product_data/channel_archive/telegram_history_inventory_matching_report_2026-05-12.json`
  является локальным read-only report (отчетом только для чтения), который можно
  учитывать как пример безопасного output (вывода) вне git.

## Термины

### Approval Draft (черновик на утверждение)

Черновик в системе Mango: web workspace (веб-рабочее место), manager bot
(бот менеджеров) или внутренняя карточка. Полезен для audit (аудита) и контроля,
но в новом MVP не заменяет native draft (нативный черновик).

### Native Draft (нативный черновик)

Настоящий черновик в поле ввода исходного Telegram-чата клиента. Это целевой UX
(пользовательский опыт) MVP.

Технически используется метод `messages.saveDraft` (сохранить черновик
сообщения), доступный через user API (пользовательский API), MTProto (протокол
Telegram-клиентов) или TDLib (официальная библиотека Telegram-клиента). Bot API
(API ботов) не умеет создавать такой persistent native draft (постоянный
нативный черновик).

### TelegramNativeDraftWriter (сервис записи нативных черновиков Telegram)

Отдельный минимальный сервис, работающий с TDLib user session (пользовательской
сессией TDLib). Он не генерирует текст, не принимает произвольные agent commands
(агентные команды), не отправляет сообщения и не читает лишние чаты. Он получает
только строго типизированные команды:

```text
save_draft(chat_id, text, reply_to_message_id?)
clear_draft(chat_id)
get_draft_state(chat_id)
```

### Business Bot Runtime (runtime бизнес-бота)

Официальный контур Telegram Business connected bot (подключенного бизнес-бота
Telegram): принимает business updates (бизнес-обновления), строит
`ChannelMessage`, сохраняет audit (аудит) и дает будущую официальную отправку
через `business_connection_id`, но в MVP основным UX отправки является ручная
отправка менеджером из native draft (нативного черновика).

## Целевая Архитектура MVP

```text
Telegram Business Account (бизнес-аккаунт Telegram)
  |
  | webhook (вебхук): business_connection / business_message / edits / deletes
  |                  (бизнес-соединение / бизнес-сообщение / изменения / удаления)
  v
TelegramBusinessWebhook (приемник вебхуков Telegram Business)
  |
  v
TelegramBusinessAdapter (адаптер Telegram Business)
  -> ChannelMessage (канальное сообщение)
  -> ChannelSQLiteStore / channel archive (SQLite-хранилище каналов / архив канала)
  -> identity matching (сопоставление личности клиента)
  -> Unified Customer Timeline (единая лента клиента)
  -> draft pipeline (конвейер черновиков)
  -> ChannelDraftPreview / ChannelDraftRecord(status=needs_review)
     (предпросмотр/запись канального черновика со статусом "нужна проверка")
  -> TelegramNativeDraftIntent (намерение записать нативный черновик)
  -> TelegramNativeDraftWriter (сервис записи нативных черновиков)
  -> messages.saveDraft (сохранить черновик в Telegram)
  -> менеджер вручную отправляет текст в обычном Telegram
```

Отдельный future path (будущий путь), не основной MVP:

```text
ControlledTelegramBusinessSender (контролируемый отправитель Telegram Business)
  -> sendMessage(business_connection_id, chat_id, approved_text)
```

Этот путь можно реализовать как fallback (резервный путь) или следующий этап, но
MVP считается готовым только после работающей записи native draft (нативного
черновика) через TDLib.

## Исполнительский Протокол Для Codex

Это ТЗ пишется для Codex (исполнителя в этом репозитории). При реализации Codex
может привлечь до 6 subagents (субагентов) с высоким reasoning effort
(усилием рассуждения), но остается ответственным за итоговую интеграцию.

Рекомендуемые subagents (субагенты):

1. `architecture explorer` (архитектурный исследователь): сверить границы
   `ChannelMessage -> draft -> native draft`.
2. `telegram/tdlib explorer` (исследователь Telegram/TDLib): проверить TDLib API
   shape (форму API) и ограничения `messages.saveDraft`.
3. `security reviewer` (ревьюер безопасности): проверить session storage
   (хранение сессии), 2FA, secrets (секреты), kill switch (аварийное отключение).
4. `storage worker` (исполнитель хранения): реализовать idempotent records
   (идемпотентные записи) и audit log (журнал аудита).
5. `tests worker` (исполнитель тестов): покрыть no-send, conflict policy
   (политику конфликтов), redaction (редакцию секретов), duplicate handling
   (обработку дублей).
6. `business workflow reviewer` (ревьюер бизнес-процесса): проверить, что UX
   реально помогает менеджеру и не создает лишних действий.

Codex не должен завершать реализационный turn (ход работы), пока не достигнуты
все deliverables (результаты) текущего work package (рабочего пакета), либо пока
не зафиксирован конкретный blocker (блокер), который нельзя обойти без доступа
или решения владельца.

Global Definition of Done (глобальный критерий готовности) для реализации MVP:

- все новые Telegram/TDLib side effects (побочные эффекты) вынесены за mockable
  transport (мокаемый транспорт);
- по умолчанию `CHANNEL_TELEGRAM_LIVE_SEND_ENABLED=false` и
  `CHANNEL_TELEGRAM_NATIVE_DRAFTS_ENABLED=false`;
- TDLib public API (публичный API TDLib-сервиса) содержит только `save_draft`,
  `clear_draft`, `get_draft_state`;
- нет `sendMessage`, raw command execution (выполнения сырых команд), arbitrary
  TDLib method proxy (прокси произвольных методов TDLib), batch operations
  (массовых операций) и outreach loops (циклов исходящих касаний);
- секреты не попадают в snapshot/report/audit/html/json (снимки/отчеты/аудит);
- есть kill switch (аварийное отключение) для native draft и controlled send;
- дубли webhook/callback/TDLib retry (повторы вебхука/callback/TDLib) не создают
  дубли сообщений, черновиков, отправок или audit events (аудит-событий);
- focused tests (фокусные тесты) и full channel smoke (полный smoke-тест
  каналов) зеленые;
- финальный отчет Codex содержит changed files (измененные файлы), test command
  (команду тестов), safety invariants (инварианты безопасности), manual WP0
  steps (ручные шаги WP0) и known limitations (известные ограничения).

## MVP Work Packages (рабочие пакеты MVP)

### WP0. Preflight (предпроверка доступов и границ)

Цель: подготовить ручные действия и секреты без записи их в git или чат.

Результаты:

- чеклист BotFather / Telegram Business для business bot (бизнес-бота);
- чеклист my.telegram.org для `TDLIB_API_ID` / `TDLIB_API_HASH`;
- чеклист 2FA (двухфакторной защиты) основного аккаунта;
- список минимальных прав business bot (бизнес-бота);
- список запрещенных прав business bot (бизнес-бота);
- локальный путь для encrypted TDLib database (зашифрованной базы TDLib);
- политика хранения секретов: `.env` или secret manager (менеджер секретов),
  без коммитов и без вставки токенов в чат.

Definition of done (критерий готовности):

- документирован manual setup checklist (ручной чеклист настройки);
- все секреты перечислены как имена переменных, но без значений;
- явно указано, что основной номер не логинится в TDLib до отдельного ручного
  подтверждения владельца;
- есть план теста сначала на test account (тестовом аккаунте), не на основном
  номере.

### WP1. Telegram Business Runtime Skeleton (скелет runtime бизнес-бота)

Цель: принять официальные Telegram Business updates (обновления Telegram
Business) и превратить их в существующий channel foundation (канальный фундамент).

Новые файлы:

```text
src/mango_mvp/channels/telegram_business_runtime.py
tests/test_channels_telegram_business_runtime.py
```

Результаты:

- `TelegramBusinessConnectionRecord` (запись бизнес-соединения);
- `TelegramBusinessUpdateRecord` (запись бизнес-обновления);
- parser (парсер) для `business_connection`, `business_message`,
  `edited_business_message`, `deleted_business_messages`;
- idempotency key (идемпотентный ключ) для каждого update (обновления);
- safe projection (безопасная проекция) без токенов и лишнего raw payload
  (сырого payload);
- bridge (мост) к существующему `TelegramReadOnlyAdapter`;
- safety contract (контракт безопасности): no network calls (нет сетевых
  вызовов), no live send (нет реальной отправки), no CRM/Tallanto writes
  (нет записей в CRM/Tallanto).

Definition of done (критерий готовности):

- тесты покрывают active/disabled connection (активное/отключенное соединение);
- тесты покрывают message/edit/delete updates (сообщение/изменение/удаление);
- повторный update (обновление) не создает дубль;
- Telegram ID хранятся как строки или 64-bit-safe values (64-битно безопасные
  значения);
- нет live Telegram API calls (реальных вызовов Telegram API).

### WP2. Native Draft Contracts (контракты нативных черновиков)

Цель: добавить внутренний слой команд для TDLib без реального логина и без
сетевых вызовов.

Новые файлы:

```text
src/mango_mvp/channels/telegram_native_draft.py
tests/test_channels_telegram_native_draft.py
```

Результаты:

- `TelegramNativeDraftIntent` (намерение нативного черновика);
- `TelegramNativeDraftState` (состояние нативного черновика);
- `TelegramNativeDraftResult` (результат операции с черновиком);
- `TelegramNativeDraftStore` или временная SQLite/table abstraction
  (абстракция хранения), если нужно для idempotency/audit;
- `NativeDraftOrchestrator` (оркестратор нативных черновиков), который принимает
  существующий `ChannelDraftPreview` / `ChannelDraftRecord` и создает
  `TelegramNativeDraftIntent`;
- stable idempotency key (стабильный идемпотентный ключ);
- validation (валидация) `chat_id`, `text`, `reply_to_message_id`;
- explicit no-send safety contract (явный контракт без отправки);
- redaction (редакция) токенов, телефонов, локальных путей в report payload
  (отчетных данных).

Definition of done (критерий готовности):

- можно создать intent (намерение), сериализовать его и получить стабильный key
  (ключ);
- нельзя создать intent (намерение) с пустым текстом, слишком длинным текстом
  или неизвестным operation (операцией);
- нет метода `sendMessage`, `send`, `forward`, `delete_chat_history` или
  произвольного raw TDLib command (сырой команды TDLib);
- тесты доказывают, что сервисный контракт не содержит auto-send (автоотправки).
- native draft write (запись нативного черновика) явно помечен как projection
  (проекция), а не approval (утверждение) и не send (отправка).

### WP3. TDLib Adapter Interface (интерфейс адаптера TDLib)

Цель: подготовить production-shaped interface (интерфейс формы production), но
сначала с fake implementation (фейковой реализацией), чтобы тесты не требовали
реального Telegram.

Новые сущности:

```text
TelegramNativeDraftClientProtocol (протокол клиента нативных черновиков)
FakeTelegramNativeDraftClient (фейковый клиент нативных черновиков)
TDLibTelegramNativeDraftClient (будущая TDLib-реализация)
```

Результаты:

- `save_draft(intent)` (сохранить черновик);
- `clear_draft(chat_id, reason)` (очистить черновик);
- `get_draft_state(chat_id)` (получить состояние черновика);
- `resolve_chat_ref(channel_message)` (сопоставить `ChannelMessage` с TDLib chat);
- fake client (фейковый клиент) для unit tests (модульных тестов);
- TDLib implementation stub (заглушка реализации TDLib), которая явно требует
  manual enable flag (ручной флаг включения) и не работает без конфигурации.

Definition of done (критерий готовности):

- fake client полностью покрыт тестами;
- TDLib stub не делает сеть и не логинится сам;
- попытка использовать TDLib без `CHANNEL_TELEGRAM_NATIVE_DRAFTS_ENABLED=true`
  возвращает blocked result (заблокированный результат);
- chat resolver (резолвер чата) не предполагает слепо, что Bot API `chat_id`
  всегда равен TDLib `chat_id`; это проверяется и логируется как mapping
  (сопоставление).
- в public API (публичном API) невозможно вызвать отправку даже через reflection
  style (рефлексивный стиль), raw payload (сырой payload) или произвольное имя
  TDLib-метода.

### WP4. Draft Pipeline -> Native Draft (конвейер черновиков в нативный черновик)

Цель: после входящего сообщения и построения `ChannelDraftPreview` /
`ChannelDraftRecord` записать текст в native draft (нативный черновик) Telegram.

Поведение:

```text
ChannelMessage
-> ChannelSession
-> identity matching
-> timeline context
-> ChannelDraftPreview / ChannelDraftRecord(status=needs_review)
-> TelegramNativeDraftIntent
-> save_draft
-> audit event: native_draft_saved
```

Результаты:

- функция `build_native_draft_intent_from_channel_draft`;
- поле/запись `last_written_hash` (хэш последнего записанного текста);
- поле/запись `native_draft_written_at` (время записи нативного черновика);
- поле/запись `native_draft_state` (состояние нативного черновика);
- audit event (событие аудита) для `native_draft_requested`,
  `native_draft_saved`, `native_draft_blocked`, `native_draft_conflict`;
- status mapping (сопоставление статусов) между `ChannelDraftRecord` и native
  draft result (результатом нативного черновика);
- если identity ambiguous/unmatched (личность неоднозначна/не сопоставлена),
  native draft все равно может быть создан, но draft metadata (метаданные
  черновика) должны содержать warning flags (предупреждающие флаги);
- если text includes commercial risk (коммерческий риск), черновик можно
  сохранить, но с флагом `requires_manager_review`.

Definition of done (критерий готовности):

- входящий Telegram Business message (бизнес-сообщение Telegram) проходит до
  `TelegramNativeDraftIntent`;
- fake native draft client сохраняет draft state (состояние черновика);
- повторная обработка того же сообщения не создает новый черновик без изменения
  текста;
- новый входящий message (сообщение) инвалидирует старый intent (намерение);
- все действия пишутся в channel history (историю канала) или отдельный audit
  record (аудит-запись).
- точный текст, записанный в Telegram draft (черновик Telegram), хранится в
  защищенном channel/draft store (хранилище каналов/черновиков), а обычные логи
  могут хранить hash/length (хэш/длину).

### WP5. Native Draft Conflict Policy (политика конфликтов нативных черновиков)

Цель: не перезаписывать ручную работу менеджера.

Правила:

- never overwrite non-empty manager draft (не перезаписывать непустой черновик
  менеджера);
- если current draft (текущий черновик) отличается от last_saved_by_mango
  (последнего сохраненного Mango), считать его manager-owned (черновиком
  менеджера);
- если пришло новое входящее сообщение, старый draft (черновик) получает статус
  `needs_rebuild`;
- если менеджер отправил сообщение вручную, draft (черновик) получает статус
  `superseded_by_manual_send`, если это можно определить;
- clear_draft (очистка черновика) разрешена только для неизмененного черновика,
  который Mango сам ранее сохранил;
- no auto clear (нет автоматической очистки) пользовательского текста.

Definition of done (критерий готовности):

- тесты покрывают empty draft (пустой черновик), unchanged Mango draft
  (неизмененный черновик Mango), modified manager draft (измененный черновик
  менеджера), new inbound invalidation (инвалидацию новым входящим сообщением);
- конфликт не теряет текст менеджера;
- конфликт явно виден в audit log (журнале аудита);
- повторная запись того же текста идемпотентна.

### WP6. Security Hardening (усиление безопасности)

Цель: TDLib user session (пользовательская сессия TDLib) безопасна настолько,
насколько это возможно для MVP.

Результаты:

- env contract (контракт переменных окружения):
  - `CHANNEL_TELEGRAM_NATIVE_DRAFTS_ENABLED`;
  - `TDLIB_API_ID`;
  - `TDLIB_API_HASH`;
  - `TDLIB_DATABASE_ENCRYPTION_KEY`;
  - `TDLIB_PHONE_NUMBER`;
  - `TDLIB_DATABASE_DIR`;
  - `CHANNEL_TELEGRAM_NATIVE_DRAFT_ALLOWED_CHAT_IDS`;
  - `CHANNEL_TELEGRAM_NATIVE_DRAFT_KILL_SWITCH`;
- guard (защита), запрещающий TDLib database dir (папку базы TDLib) внутри git
  tracked paths (путей под git), `stable_runtime`, repo root (корня репозитория)
  и публичных temp paths (временных путей);
- no secrets in logs (секреты не попадают в логи);
- no session dump in reports (сессия не попадает в отчеты);
- explicit test-account-first gate (явное требование сначала тестового аккаунта);
- kill switch behavior (поведение аварийного отключения).

Definition of done (критерий готовности):

- тесты доказывают, что секреты редактируются;
- TDLib нельзя включить без feature flag (флага функции);
- kill switch блокирует `save_draft`;
- unsafe paths (опасные пути) отклоняются;
- в git не появляется `.env`, TDLib database (база TDLib), session files
  (файлы сессии), raw Telegram exports (сырые экспорты Telegram).

### WP7. Manager UX (пользовательский опыт менеджера)

Цель: менеджер получает черновик там, где работает, и понимает его статус.

MVP UX:

- native draft (нативный черновик) появляется в исходном Telegram-чате;
- дополнительная manager notification (уведомление менеджера) допустима через
  manager bot (бот менеджеров) или workspace (рабочее место), но не заменяет
  native draft;
- карточка менеджера должна показывать:
  - клиент matched/ambiguous/unmatched (сопоставлен/неоднозначен/не сопоставлен);
  - source refs (ссылки на источники);
  - exact draft text (точный текст черновика);
  - warnings (предупреждения);
  - native draft status (статус нативного черновика).

Definition of done (критерий готовности):

- есть JSON/workspace summary (сводка рабочего места) для менеджера;
- статус `native_draft_saved` виден в результате;
- если native draft blocked/conflict (заблокирован/конфликтует), менеджер видит
  причину и может ответить вручную;
- нет кнопки, которая отправляет сообщение клиенту без Telegram/manual review
  (ручной проверки в Telegram).

### WP7a. Manual Send Reconciliation (сверка ручной отправки)

Цель: MVP не отправляет сообщение сам, но должен понимать, когда менеджер
вручную отправил ответ из Telegram, если это видно через incoming/outgoing
business updates (входящие/исходящие бизнес-обновления) или исторический sync
(синхронизацию).

Результаты:

- если ручной ответ соответствует последнему `last_written_hash`, draft
  (черновик) получает статус или reason (причину) `manual_send_observed`;
- если менеджер отправил другой текст, draft (черновик) получает reason
  `manager_sent_modified_text`;
- если событие отправки недоступно через API, система не делает ложный вывод, а
  оставляет статус `native_draft_saved`;
- AI (ИИ) не переписывает manager-edited text (текст, измененный менеджером).

Definition of done (критерий готовности):

- тесты покрывают observed manual send (замеченную ручную отправку), modified
  manual send (измененную ручную отправку) и unknown send state (неизвестное
  состояние отправки);
- reconciliation (сверка) идемпотентна;
- никакая сверка не вызывает Telegram send (отправку Telegram).

### WP8. Test Suite And Smoke (тесты и smoke-проверка)

Цель: Codex не завершает реализацию, пока новый MVP не проверен.

Обязательные тесты:

- business update parsing (парсинг бизнес-обновлений);
- deleted/edited business events (удаленные/измененные бизнес-события);
- draft pipeline to native draft intent (конвейер до намерения нативного
  черновика);
- fake native draft save/get/clear (фейковое сохранение/чтение/очистка);
- no send endpoint (нет endpoint-а отправки);
- no auto-send (нет автоотправки);
- secret redaction (редакция секретов);
- unsafe TDLib path rejection (отклонение опасного пути TDLib);
- kill switch (аварийное отключение);
- conflict policy (политика конфликтов);
- idempotency (идемпотентность);
- no CRM/Tallanto writes (нет записей в CRM/Tallanto);
- no stable_runtime writes (нет записей в stable_runtime);
- historical Telegram tests remain green (исторические Telegram-тесты остаются
  зелеными).

Команда проверки:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_channels_*.py tests/test_customer_timeline_channel_preview_from_pack.py
```

Definition of done (критерий готовности):

- все channel tests (тесты каналов) зеленые;
- новые tests (тесты) покрывают MVP native draft path (путь нативного черновика);
- итоговый отчет Codex перечисляет измененные файлы, проверку, известные
  ограничения и ручные preflight steps (шаги предпроверки).

## Safety Contract (контракт безопасности)

Обязательные инварианты MVP:

- native draft is allowed (нативный черновик разрешен);
- live send is not allowed (реальная отправка не разрешена);
- no auto-send (нет автоотправки);
- no outbound/spam campaigns (нет исходящих спам-кампаний);
- no OpenClaw access to main Telegram user session (OpenClaw не получает доступ
  к пользовательской сессии основного Telegram);
- TDLib service has no send endpoint (TDLib-сервис не имеет endpoint-а отправки);
- TDLib service has no arbitrary command endpoint (TDLib-сервис не имеет
  endpoint-а произвольных команд);
- no generic TDLib RPC endpoint (нет универсального endpoint-а TDLib RPC);
- native draft write is not approval and not send (запись нативного черновика
  не является утверждением и не является отправкой);
- Mango audit stores exact text written to Telegram draft (аудит Mango хранит
  точный текст, записанный в черновик Telegram, в защищенном хранилище);
- no CRM/Tallanto writes from Telegram adapter (адаптер Telegram не пишет в
  CRM/Tallanto);
- no ASR/R+A from Telegram adapter (адаптер Telegram не запускает ASR/R+A);
- no legacy RAG (нет старой RAG-системы);
- all draft actions require idempotency (все действия с черновиками требуют
  идемпотентности);
- all AI facts have source refs or confidence flags (все факты ИИ имеют ссылки
  на источники или флаги уверенности);
- manager manual fallback is always available (ручной резервный сценарий для
  менеджера всегда доступен);
- kill switch is tested before pilot (аварийное отключение проверено до пилота).

## Environment (переменные окружения)

```text
CHANNEL_TELEGRAM_BUSINESS_BOT_TOKEN
CHANNEL_TELEGRAM_MANAGER_BOT_TOKEN
CHANNEL_TELEGRAM_WEBHOOK_SECRET
CHANNEL_TELEGRAM_LIVE_SEND_ENABLED=false
CHANNEL_TELEGRAM_DRAFTS_ENABLED=true
CHANNEL_TELEGRAM_NATIVE_DRAFTS_ENABLED=false
CHANNEL_TELEGRAM_NATIVE_DRAFT_KILL_SWITCH=true
CHANNEL_TELEGRAM_ALLOWED_BUSINESS_CONNECTION_IDS
CHANNEL_TELEGRAM_ALLOWED_MANAGER_IDS
CHANNEL_TELEGRAM_NATIVE_DRAFT_ALLOWED_CHAT_IDS
TDLIB_API_ID
TDLIB_API_HASH
TDLIB_DATABASE_ENCRYPTION_KEY
TDLIB_PHONE_NUMBER
TDLIB_DATABASE_DIR
```

Правила:

- не коммитить `.env`;
- не вставлять токены, api hash (API hash), 2FA password (пароль 2FA) или
  TDLib encryption key (ключ шифрования TDLib) в чат;
- production secrets (боевые секреты) хранить в secret manager (менеджере
  секретов);
- основной аккаунт должен иметь 2FA;
- сначала test account (тестовый аккаунт), только потом основной номер;
- TDLib database (база TDLib) хранится вне git и вне `stable_runtime`.

## Аудит Трех Ролей После Нового Требования

### Строгий IT-архитектор / Программист

Согласен включить native draft (нативный черновик) в MVP при условиях:

- TDLib изолирован в `TelegramNativeDraftWriter`;
- API сервиса не содержит отправки;
- используется fake client (фейковый клиент) для тестов;
- реальный TDLib включается только feature flag (флагом функции) и после
  manual preflight (ручной предпроверки);
- conflict policy (политика конфликтов) реализована до подключения основного
  номера.

### Опытный Бизнесмен / Руководитель Продаж

Согласен включить native draft (нативный черновик) в MVP, потому что это снижает
сопротивление менеджеров: они остаются в привычном Telegram-чате. Но отправка
остается ручной, чтобы не рисковать основным номером и качеством коммуникации.

### Позиция Mango Analyse

Telegram остается adapter layer (адаптерным слоем), а не отдельным продуктом:

```text
Telegram -> ChannelMessage -> identity -> timeline -> draft -> native draft
```

Нативный черновик - это delivery surface (поверхность доставки черновика), а не
самостоятельная логика продаж. Источник правды остается в channel store
(хранилище каналов), audit log (журнале аудита) и Unified Customer Timeline
(единой ленте клиента).

### Единая Позиция

Все роли согласны:

1. Business Bot API (API бизнес-ботов) остается официальным контуром приема.
2. Native draft (нативный черновик) входит в MVP.
3. `TelegramNativeDraftWriter` делает только save/get/clear draft
   (сохранить/прочитать/очистить черновик).
4. Никакой auto-send (автоотправки).
5. Никаких OpenClaw/userbot shortcuts (коротких путей через OpenClaw/userbot) на
   основном номере.
6. Codex не заканчивает реализацию, пока не сделаны contracts (контракты),
   tests (тесты), fake client (фейковый клиент), safety gates (защитные
   проверки), docs (документация) и status report (отчет о статусе).

## Что Не Делаем В MVP

- Не отправляем сообщения автоматически.
- Не даем TDLib-сервису endpoint (endpoint) отправки.
- Не даем TDLib-сервису произвольный raw command endpoint (endpoint сырых
  команд).
- Не используем TDLib-сессию ни для чего, кроме save/read/clear native draft
  (сохранения/чтения/очистки нативного черновика) через изолированный
  `TelegramNativeDraftWriter`.
- Не подключаем OpenClaw к пользовательской сессии.
- Не делаем массовый outbound (исходящие рассылки).
- Не пишем в CRM/Tallanto из Telegram-контура.
- Не запускаем ASR/R+A.
- Не используем legacy RAG (старую RAG-систему).
- Не коммитим raw Telegram exports (сырые Telegram-экспорты), TDLib database
  (базу TDLib), session files (файлы сессии), tokens (токены), `.env`.

## Первый Реализационный Срез

Codex должен начать с безопасного offline implementation (офлайн-реализации):

1. Создать `telegram_business_runtime.py` и тесты.
2. Создать `telegram_native_draft.py` и тесты.
3. Подключить fake native draft client (фейковый клиент нативных черновиков).
4. Провести один end-to-end dry run (сквозной сухой прогон):

```text
business_message fixture (тестовое бизнес-сообщение)
-> ChannelMessage
-> ChannelDraftPreview / ChannelDraftRecord
-> TelegramNativeDraftIntent
-> FakeTelegramNativeDraftClient.save_draft
-> audit result: native_draft_saved
```

5. Запустить focused tests (фокусные тесты).
6. Обновить docs (документацию) и итоговый report (отчет).

Go/No-Go (условия запуска на реальном аккаунте):

- No-go (нельзя запускать), если нет 2FA, encrypted TDLib database
  (зашифрованной базы TDLib), kill switch (аварийного отключения), allowlist
  (списка разрешенных чатов/менеджеров), no-send tests (тестов отсутствия
  отправки), conflict policy tests (тестов политики конфликтов) и manual session
  revoke runbook (инструкции отзыва сессии).
- Go only for test account first (сначала только тестовый аккаунт).
- Основной номер подключается только после отдельного явного разрешения владельца
  и успешного тестового пилота.

Реальный TDLib login (логин TDLib) и реальная запись черновика в Telegram
разрешены только отдельным шагом после offline tests (офлайн-тестов), manual
preflight (ручной предпроверки), test account (тестового аккаунта) и явного
разрешения владельца.

## Official References (официальные источники)

- Telegram Bot API (API ботов Telegram): поля update (обновлений) включают
  `business_connection`, `business_message`, `edited_business_message`,
  `deleted_business_messages`: https://core.telegram.org/bots/api
- `BusinessBotRights.can_reply` (право бизнес-бота отвечать) ограничено
  приватными чатами с входящими сообщениями за последние 24 часа:
  https://core.telegram.org/bots/api
- `sendMessage` (отправить сообщение) и другие методы поддерживают
  `business_connection_id` (идентификатор бизнес-соединения):
  https://core.telegram.org/bots/api
- Connected business bots (подключенные бизнес-боты) обрабатывают и отвечают на
  сообщения от имени бизнеса:
  https://core.telegram.org/api/bots/connected-business-bots
- TDLib (официальная библиотека Telegram-клиента) является fully functional
  client library (полнофункциональной клиентской библиотекой):
  https://core.telegram.org/tdlib
- `messages.saveDraft` (сохранить черновик сообщения) является user-only
  method (методом только для пользователей):
  https://core.telegram.org/method/messages.saveDraft
- Telegram API client libraries (клиентские библиотеки Telegram API)
  мониторятся для предотвращения abuse (злоупотреблений):
  https://core.telegram.org/api/obtaining_api_id
- Telegram API Terms (условия Telegram API) ограничивают использование Telegram
  data (данных Telegram) для AI/ML training/fine-tuning
  (обучения/дообучения AI/ML-моделей):
  https://core.telegram.org/api/terms
