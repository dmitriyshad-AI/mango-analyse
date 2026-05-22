# Реестр архитектурных рисков

## 1. P0 риски

### R-001. Автоответ по P0-теме

Риск: бот автономно отвечает по возврату, жалобе, суду, спорной оплате или претензии.

Последствие: вред клиенту, бренду, юридический риск.

Контроль:

- P0 route всегда `manager_only`;
- смешанный вопрос с P0 целиком уходит менеджеру;
- P0 fast tests;
- daily semantic review.

### R-002. Смешение Фотон/УНПК

Риск: бот Фотона даёт факты УНПК или наоборот.

Последствие: неверная консультация, потеря доверия.

Контроль:

- active_brand required;
- cross-brand post-filter;
- отдельные bot tokens;
- отдельные KB scopes;
- tests by brand.

### R-003. Выдуманный факт

Риск: бот называет цену, дату, наличие мест, скидку, срок или обещание без проверенного факта.

Контроль:

- fact requirement для автономности;
- `client_safe_fact_verified`;
- missing_facts -> no autonomous answer;
- gold-ответы как стиль, не как источник произвольных фактов.

### R-004. Раскрытие внутреннего контекста

Риск: бот пишет клиенту “в AMO у вас...”, “Tallanto id...”, “у вас 85 звонков”.

Контроль:

- context can be used for understanding only;
- client-visible text не содержит source ids;
- post-filter на CRM/Tallanto/AMO/GPT/Codex/source_id.

### R-005. Live-write shortcut

Риск: кто-то добавляет запись в AMO/Tallanto/CRM или live-send в обход gates.

Контроль:

- no live write by default;
- explicit approval token;
- snapshot/rollback/readback для CRM;
- code review на live-команды;
- tests: write flags false.

## 2. P1 риски

### R-006. Бот безопасен, но бесполезен

Риск: ответы формально безопасны, но сотрудникам не помогают.

Контроль:

- feedback verdict;
- useful rate;
- employee corrected answer;
- gold-answer quality rules;
- report “шаблонные ответы”.

### R-007. Нет постоянного журнала пилота

Риск: нельзя понять, что произошло и почему.

Контроль:

- local SQLite store;
- daily report;
- decision log;
- used facts;
- route and flags.

### R-008. Повторный запрос уже известных данных

Риск: клиент найден, но бот снова спрашивает имя, телефон, класс, бренд.

Контроль:

- known_client_fields;
- asked_known_data_again flag;
- review queue;
- tests.

### R-009. Старая база знаний используется после обновления

Риск: процесс запущен со старым snapshot.

Контроль:

- log `knowledge_release_id`;
- active symlink/config;
- startup check;
- semantic_pass required;
- rollback procedure.

### R-010. Customer timeline принят за истину слишком рано

Риск: семейные телефоны, дубли Tallanto и несколько детей дают неверный контекст.

Контроль:

- timeline read-only;
- confidence;
- requires_manager_review;
- no auto answer on ambiguous identity.

## 3. P2 риски

### R-011. Преждевременный SaaS

Риск: команда уйдёт в роли, биллинг, облако, админку до доказанной внутренней пользы.

Контроль:

- readiness gates;
- no SaaS before internal production;
- focus on Telegram pilot metrics.

### R-012. Telegram-specific ядро

Риск: все правила окажутся внутри Telegram-кода.

Контроль:

- `AiEmployeeCore`;
- `DecisionEnvelope`;
- channel adapters без бизнес-логики;
- одинаковые tests для Telegram/web-chat.

### R-013. SQLite задержится слишком долго

Риск: для hosted/multi-user режима SQLite станет узким местом.

Контроль:

- SQLite for appliance only;
- repository interfaces;
- migration point after external pilot.

### R-014. Непрозрачное обновление KB

Риск: база знаний обновилась, но непонятно, что изменилось.

Контроль:

- diff report;
- semantic review;
- release notes;
- owner approval;
- rollback.

## 4. P3 риски

### R-015. Слишком много документов и скриптов

Риск: новый диалог берёт старый документ как актуальный.

Контроль:

- status map;
- current state update;
- docs archive labels.

### R-016. Локальные артефакты попадают в git

Риск: логи, секреты, raw payload или персональные данные попадают в репозиторий.

Контроль:

- `.gitignore`;
- pre-commit check;
- audit pack redaction.

## 5. Анти-паттерны

1. “Добавим SaaS, а потом разберёмся с пилотом.”
2. “Telegram работает, значит ядро готово.”
3. “База знаний прошла формальную проверку, значит можно автоответы.”
4. “Tenant_id есть, значит изоляция есть.”
5. “Клиент спросил смешанный вопрос, ответим на безопасную часть автономно.”
6. “Менеджер потом исправит, можно отправлять.”
7. “Gold-ответ можно копировать дословно.”
8. “CRM/Tallanto read-only контекст можно раскрыть клиенту.”

## 6. Главная рекомендация

Каждый риск должен переводиться в одно из трёх:

- тест;
- semantic gate;
- правило в базе знаний или матрице автономности.

Иначе риск вернётся.
