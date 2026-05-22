# Спецификация хранения диалогов, решений, флагов и отзывов

## 1. Зачем нужно постоянное хранение

Сейчас часть состояния Telegram-пилота живёт в памяти процесса и логах. Для первого теста это допустимо, но для внутреннего ИИ-сотрудника и будущего продукта нужно хранить:

- что написал клиент;
- какой ответ дал бот;
- почему был выбран route;
- какие факты использовались;
- какой бренд был активен;
- какие риски сработали;
- что сотрудник исправил;
- помог ли ответ.

Без этого невозможно улучшать бота системно.

**Важно: сейчас реализуется только минимальный slice из раздела 8. Остальная схема - целевая спецификация на будущее, а не текущее задание на разработку.**

## 2. Где хранить

### 2.1. Внутренний пилот

Рекомендуется SQLite в product root:

```text
product_data/db/ai_employee_pilot.sqlite
```

Или отдельная channel DB:

```text
product_data/db/channel_runtime.sqlite
```

Запрещено:

- хранить это в `stable_runtime`;
- писать в runtime DB звонков;
- класть токены в эту базу;
- сохранять raw Telegram update по умолчанию.

### 2.2. Будущий client-hosted продукт

Для одной организации SQLite пока допустим:

- проще backup;
- проще перенос;
- меньше администрирования;
- подходит для MacBook/сервер клиента.

Для hosted SaaS позже потребуется PostgreSQL или другой серверный слой.

## 3. Основные таблицы

### 3.1. `conversation_sessions`

Одна строка на диалог/чат.

Поля:

- `session_id`
- `tenant_id`
- `brand_id`
- `channel`
- `channel_thread_id`
- `channel_user_ref_hash`
- `customer_id`
- `identity_status`
- `manager_owner_id`
- `status`
- `created_at`
- `updated_at`
- `last_message_at`
- `metadata_json`

Важно: внешний `channel_user_id` хранить либо хэшированным, либо в закрытой локальной БД, не в audit pack.

### 3.2. `conversation_messages`

Входящие и исходящие сообщения.

Поля:

- `message_id`
- `session_id`
- `tenant_id`
- `brand_id`
- `channel`
- `direction`
- `message_text`
- `message_text_redacted`
- `received_at`
- `idempotency_key`
- `source_message_id`
- `attachment_count`
- `contains_personal_data`
- `raw_payload_ref`
- `created_at`

Правило: `raw_payload_ref` может ссылаться на закрытый локальный файл, но сам raw payload не попадает в открытые отчёты.

### 3.3. `context_snapshots`

Контекст, который был доступен на момент ответа.

Поля:

- `context_id`
- `session_id`
- `message_id`
- `tenant_id`
- `brand_id`
- `knowledge_release_id`
- `knowledge_snapshot_sha256`
- `semantic_pass`
- `active_brand`
- `customer_found`
- `known_client_fields_json`
- `known_dialog_fields_json`
- `amo_context_status`
- `tallanto_context_status`
- `timeline_context_status`
- `facts_used_json`
- `facts_missing_json`
- `gold_rules_used_json`
- `context_flags_json`
- `created_at`

Важно: не сохранять полные AMO/Tallanto payload в открытом виде. Хранить только summary/status/ref.

### 3.4. `bot_decisions`

Решение ядра по конкретному сообщению.

Поля:

- `decision_id`
- `session_id`
- `message_id`
- `context_id`
- `tenant_id`
- `brand_id`
- `route`
- `topic_id`
- `secondary_topic_ids_json`
- `autonomy_allowed`
- `autonomy_blockers_json`
- `safety_flags_json`
- `semantic_flags_json`
- `post_filter_flags_json`
- `asked_known_data_again`
- `latency_ms`
- `model_name`
- `reasoning_effort`
- `prompt_version`
- `decision_status`
- `created_at`

Route values:

- `bot_answer_self_for_pilot`
- `draft_for_manager`
- `manager_only`
- `blocked`
- `fallback`

### 3.5. `bot_answers`

Сгенерированные ответы.

Поля:

- `answer_id`
- `decision_id`
- `session_id`
- `message_id`
- `tenant_id`
- `brand_id`
- `answer_text`
- `answer_text_redacted`
- `answer_kind`
- `sent_to_client`
- `sent_to_manager`
- `client_send_allowed`
- `client_send_executed`
- `manager_handoff_executed`
- `template_like_score`
- `direct_answer_score`
- `created_at`

В пилоте `client_send_executed` может быть true только если текущие правила явно разрешают автономный ответ. P0 всегда запрещает.

### 3.6. `manager_feedback`

Оценка сотрудника или РОПа.

Поля:

- `feedback_id`
- `decision_id`
- `answer_id`
- `reviewer_id`
- `reviewer_role`
- `verdict`
- `corrected_answer_text`
- `problem_type`
- `comment`
- `next_action`
- `should_update_kb`
- `should_add_test`
- `should_update_prompt`
- `created_at`

Verdict:

- `helpful`
- `helpful_with_minor_edit`
- `rewritten_by_manager`
- `useless`
- `dangerous`
- `wrong_fact`
- `wrong_brand`
- `needs_kb_fact`
- `needs_new_test`

### 3.7. `pilot_metrics_daily`

Агрегат по дню и бренду.

Поля:

- `date`
- `tenant_id`
- `brand_id`
- `messages_in`
- `answers_out`
- `autonomous_answers`
- `manager_only`
- `fallbacks`
- `p0_blocked`
- `with_crm_context`
- `with_tallanto_context`
- `asked_known_data_again`
- `helpful_count`
- `dangerous_count`
- `median_latency_ms`
- `semantic_review_queue_count`

## 4. Что можно хранить в локальной пилотной БД

Можно:

- тексты сообщений;
- ответы бота;
- route;
- flags;
- topic;
- latency;
- known fields;
- masked phone/email;
- ссылки на локальные закрытые raw-файлы;
- исправления сотрудников;
- вердикты сотрудников;
- использованные fact ids.

Но это должна быть локальная приватная БД, не audit pack.

## 5. Что нельзя хранить в открытых audit packs

Нельзя:

- Telegram token;
- API keys;
- полные raw Telegram updates;
- полные AMO/Tallanto payload;
- телефоны в открытом виде;
- email в открытом виде;
- ФИО клиента и ребёнка без необходимости;
- тексты с персональными данными;
- вложения клиентов;
- внутренние CRM id, если они не нужны для аудита.

В audit pack можно класть:

- masked phone: `+7***0033`;
- hashed user id;
- redacted text;
- route;
- safety flags;
- problem type;
- summary;
- aggregate metrics.

## 6. Маскирование персональных данных

Минимальные правила:

- телефон: оставить последние 2-4 цифры;
- email: `a***@domain.ru`;
- ФИО: заменить на роль, если имя не нужно для проверки;
- Telegram username: хэш или маска;
- AMO/Tallanto ids: только если нужны для внутреннего readback, не для публичного audit pack;
- вложения: не копировать, хранить только hash/size/type.

## 7. Как переносить в SaaS

Путь миграции:

1. Внутренний SQLite store.
2. Явная схема и миграции.
3. Backup/restore dry-run.
4. Разделение `tenant_id` и `brand_id`.
5. Экспорт без персональных данных для демо.
6. Перенос активных таблиц в PostgreSQL только при реальной multi-user нагрузке.

## 8. Минимальный первый storage-slice

Не нужно сразу строить всю БД. Первый полезный slice:

1. `conversation_sessions`
2. `conversation_messages`
3. `bot_decisions`
4. `bot_answers`
5. `manager_feedback`
6. ежедневный report builder

Этого достаточно, чтобы понять, полезен ли внутренний бот.

Всё остальное из этой спецификации подключать только после того, как минимальный slice начал ежедневно давать пользу: понятный журнал, feedback сотрудников и разбор ошибок.
