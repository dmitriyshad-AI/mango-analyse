# План реализации Transcript Quality Guardrails v2

Дата: 2026-05-09.

Связанные аудиты:

- `docs/TRANSCRIPT_QUALITY_GUARDRAILS_AUDIT_2026-05-09.md`
- `docs/TRANSCRIPT_QUALITY_ADVERSARIAL_AUDIT_2026-05-09.md`
- `stable_runtime/transcript_quality_adversarial_audit_20260509/`

## Цель

Исправить системную проблему, при которой автоответчики, недозвоны, голосовая почта и ASR-артефакты иногда проходят дальше как содержательные звонки и попадают в readiness, sales moments, Knowledge Base, ROP pack и bot seeds.

Правки должны работать:

1. Для всех будущих звонков.
2. Для уже обработанных 64 832 старых звонков без повторного ASR.

Главный принцип: не ухудшить качество. Нельзя делать грубый regex-фильтр, потому что слова `почта`, `сообщение`, `перезвон`, `недоступен`, `оставить` встречаются и в живых продажах/сервисе.

## Этап 1. Зафиксировать baseline

Перед правками зафиксировать текущую точку:

- количество `contentful`;
- количество `non_conversation`;
- распределение `sales_call`, `service_call`, `technical_call`, `existing_client_progress`;
- сколько строк в ROP попадает в `Риски потери выручки`;
- сколько bot seeds содержит цены/скидки/дедлайны/возвраты/brand-risk;
- сколько no-live строк попадает в revenue risks.

Результат: можно будет честно сравнить v1 и v2.

## Этап 2. Создать общий quality module

Создать модуль:

`src/mango_mvp/quality/non_conversation.py`

Модуль должен давать не бинарное `true/false`, а объяснимую оценку:

- `strong_no_live_marker`;
- `asr_artifact_marker`;
- `system_no_dialogue_phrase`;
- `client_turn_length`;
- `client_live_business_terms`;
- `history_live_business_terms`;
- `structured_business_fields`;
- `duration/transcript_length`;
- `manager/client speaker balance`;
- `recommended_quality_label`;
- `reason_codes`.

Возможные статусы:

- `non_conversation_high_confidence`;
- `manual_review_probable_no_live`;
- `manual_review_borderline_live_context`;
- `contentful_protected_live_dialogue`.

## Этап 3. Покрыть quality module тестами

Использовать regression dataset:

`stable_runtime/transcript_quality_adversarial_audit_20260509/regression_dataset.csv`

Проверить:

- high-confidence no-live становится no-live;
- probable no-live не применяется автоматически без review;
- borderline не применяется автоматически;
- живые `sales_call`, `service_call`, `technical_call`, `existing_client_progress` остаются contentful;
- `голосовая почта` не становится email;
- автоответчик `перезвоните позднее` не становится согласованным `next_step`.

## Этап 4. Подключить module в Analyze

Файл:

`src/mango_mvp/services/analyze.py`

Изменить:

- `_is_non_conversation()`;
- `_detect_call_type()`;
- `_candidate_next_step_action()`;
- `_detect_preferred_channel()`;
- `_normalize_analysis()`.

Правила:

1. Сильный no-live + нет живого диалога => `call_type=non_conversation`.
2. No-live не может иметь `follow_up_score > 0`.
3. No-live не может иметь `lead_priority=warm/hot`.
4. No-live не может иметь обычный `next_step`.
5. `голосовая почта` / `голосовой почтовый ящик` не может давать `preferred_channel=email`.
6. В `analysis_json.quality_flags` добавлять `quality_reason_codes`.

## Этап 5. Подключить contentful v2 в readiness

Файл:

`src/mango_mvp/insights/readiness.py`

Добавить в output:

- `contentful_v2`;
- `contentfulness_reason`;
- `non_conversation_reason_codes`;
- `asr_artifact_flag`;
- `no_live_flag`;
- `live_dialogue_evidence_level`.

Старое поле `contentful` не удалять. Новые отчеты должны использовать `contentful_v2`.

## Этап 6. Исправить pilot extraction

Файл:

`src/mango_mvp/insights/pilot_extraction.py`

Правки:

- не брать high-confidence no-live/artifact в sales moments;
- borderline не отправлять в LLM review автоматически;
- protected-live звонки не удалять;
- добавлять reason codes в output.

## Этап 7. Исправить Knowledge Base

Файл:

`src/mango_mvp/insights/knowledge_base.py`

Правки:

- `no_live_contact_or_voicemail` никогда не должен быть `revenue_leakage_risk`;
- no-live не должен быть `playbook_candidate`;
- no-live не должен попадать в bot seeds;
- добавить `ideal_answer_for_manager`;
- добавить `safe_answer_for_bot`;
- добавить bot safety flags:
  - `brand_risk_flag`;
  - `money_or_discount_flag`;
  - `installment_flag`;
  - `legal_or_refund_flag`;
  - `deadline_or_promise_flag`;
  - `personal_data_flag`.

## Этап 8. Исправить ROP pack

Файл:

`src/mango_mvp/insights/rop_validation_pack.py`

Правки:

- добавить лист `Недозвоны / non-conversation`;
- убрать no-live из `Риски потери выручки`;
- убрать no-live из `ТОП ответы`;
- убрать no-live из `Черновики для бота`;
- добавить в `Сводка` счетчики исключений и safety flags.

## Этап 9. Сделать dry-run по всем старым звонкам

Создать скрипт:

`scripts/backfill_non_conversation_quality_v2.py`

Режимы:

- `--dry-run`;
- `--write-sidecar`;
- `--apply`.

На первом запуске использовать только `--dry-run` / `--write-sidecar`.

Скрипт должен пройти по старым 64 832 звонкам и создать sidecar CSV/JSONL:

- `source_db`;
- `source_filename`;
- `old_call_type`;
- `new_call_type_candidate`;
- `old_contentful`;
- `new_contentful_candidate`;
- `quality_label`;
- `reason_codes`;
- `risk_level`;
- `recommended_action`;
- `requires_manual_review`.

На этом этапе SQLite не меняется.

## Этап 10. Проверить dry-run

Проверить:

- сколько high-confidence no-live;
- сколько borderline;
- сколько protected-live;
- какие месяцы, менеджеры и типы звонков затронуты;
- примеры до/после;
- не слишком ли агрессивно падает contentful.

Проверить вручную sample:

- 50 high-confidence;
- 50 borderline;
- 50 protected-live.

Автоматически применять можно только high-confidence при высокой точности.

## Этап 11. Staged backfill старых звонков

Backfill делать только после dry-run и проверки.

Правила:

1. Не запускать повторный ASR.
2. Не удалять старые значения.
3. Добавлять v2-значения и reason codes.
4. Применять автоматически только high-confidence.
5. Borderline оставить отдельным списком для manual review.

Что может быть обновлено:

- `analysis_json.quality_flags.call_type_v2`;
- `analysis_json.quality_flags.contentful_v2`;
- `analysis_json.quality_flags.quality_reason_codes`;
- `analysis_json.quality_flags.quality_label`;
- при необходимости derived summary fields, но только после backup.

## Этап 12. Пересобрать производные слои

После backfill:

1. readiness v2;
2. outcome linkage v2;
3. pilot sales moments v2;
4. LLM review только для новых/затронутых moments;
5. Knowledge Base v3;
6. ROP pack v2;
7. bot seed candidates v2.

## Этап 13. Сравнить v1/v2

Сделать comparison report:

- сколько no-live ушло из revenue risks;
- сколько no-live ушло из bot seeds;
- сколько bot seeds исключено из-за цен/скидок/возвратов/дедлайнов;
- сколько top answers осталось;
- сколько живых sales/service/technical звонков защищено;
- нет ли подозрительного падения contentful;
- сколько строк ушло в manual review.

## Этап 14. Включить для новых звонков

После успешной проверки включить quality module в постоянный пайплайн:

ASR -> Resolve -> Analyze -> Quality classification -> readiness/KB/ROP/bot/CRM.

## Критерии успеха

1. Недозвоны не попадают в ROP как `P0 риск выручки`.
2. Автоответчики не попадают в bot seeds.
3. Живые service/technical/sales звонки не теряются.
4. `голосовая почта` не становится email.
5. Автоответчик `перезвоните позднее` не становится согласованным next step.
6. Бот не получает цены, скидки, возвраты, дедлайны и неправильный бренд без sanitizer.
7. Старые 64 832 звонка доработаны без повторного ASR.
8. Все изменения объяснимы через reason codes.

## Что не делать

- Не править Excel вручную.
- Не применять массовую запись в SQLite без dry-run.
- Не запускать повторный ASR.
- Не заменять старые значения без v2/sidecar.
- Не использовать `ideal_answer_example` как готовый bot seed без safe-answer sanitizer.
