# Третий аудит: adversarial pre-fix validation

Дата: 2026-05-09.

Связанный базовый аудит: `docs/TRANSCRIPT_QUALITY_GUARDRAILS_AUDIT_2026-05-09.md`.

Артефакты аудита: `stable_runtime/transcript_quality_adversarial_audit_20260509/`.

## Цель

Проверить предложенные правки до изменения кода:

1. Где фильтры no-live/non-conversation могут ошибочно выкинуть живые звонки.
2. Какие no-live/ASR-artifact случаи старые маркеры еще пропускают.
3. Как безопасно backfill-ить 60k+ обработанных звонков без повторного ASR.
4. Какие regression examples нужны, чтобы после правок ничего не стало хуже.

## Метод

Аудит прошел по текущим производным данным:

- `stable_runtime/insight_readiness_report_20260507/calls_terminal_analyzed.csv`
- `stable_runtime/sales_insight_knowledge_base_20260507_v2/enriched_reviews.csv`
- `stable_runtime/rop_validation_pack_20260507_v1/rop_validation.csv`

Для рискованных строк дополнительно подтянут `transcript_text` из SQLite source DB.

Итог:

- строк readiness: 64 832;
- подтянутых transcript_text: 43 781;
- regression dataset: 90 строк.

## Главный вывод

Фильтр нельзя строить как простой regex по словам `почта`, `сообщение`, `перезвон`, `оставить`, `недоступен`.

Такие слова массово встречаются в живых и коммерчески ценных звонках:

- живые sales_call с рискованными словами: 18 954;
- живые service_call с рискованными словами: 3 053;
- живые technical_call с рискованными словами: 283;
- живые existing_client_progress с рискованными словами: 1 012.

Следовательно, правильный фильтр должен быть двухфакторным:

1. Есть сильный no-live/system marker.
2. Нет убедительного live-dialogue evidence.

Если есть живой клиентский ответ, обсуждение оплаты/курса/расписания/доступа/записи/возражений, звонок нельзя автоматически переводить в `non_conversation`.

## Новые количественные результаты

### High-confidence no-live/artifact

Найдено 5 558 contentful-звонков, которые выглядят как high-confidence no-live/artifact candidates.

Разбивка:

- `service_call`: 2 380;
- `technical_call`: 2 246;
- `sales_call`: 880;
- `existing_client_progress`: 52.

Top months:

- 2025-09: 1 324;
- 2025-10: 581;
- 2025-11: 505;
- 2025-02: 449;
- 2025-04: 319;
- 2026-02: 319;
- 2026-04: 309.

Важно: это still candidate set. Его нельзя сразу писать в DB без dry-run и выборочной проверки.

### Borderline

Найдено:

- `borderline_probable_no_live`: 75;
- `borderline_possible_false_positive`: 283.

Это звонки, где есть no-live/system marker, но также есть признаки живого контекста: например, секретарь ответил вместо клиента, менеджер оставил содержательное голосовое, или в summary подтянули историю/продукты.

Правило: borderline не backfill-ить автоматически. Только отдельный manual-review или оставить contentful до более точного правила.

### Старые маркеры слишком узкие

Старый набор markers не сработал на:

- 3 044 high-confidence no-live/artifact candidates;
- 63 borderline probable no-live;
- 220 borderline possible false positive;
- 1 007 уже помеченных `non_conversation`, где старые markers тоже не ловили текст.

Это подтверждает, что проблема не только downstream. Analyze действительно должен получить расширенный detector.

## Downstream impact

По Knowledge Base v2:

- `no_live_contact_or_voicemail`, ошибочно попавшие в `revenue_leakage_risk`: 20;
- bot-ready/needs-validation с деньгами/скидками/рассрочкой/возвратом/дедлайнами: 630;
- `ideal_answer_example` с brand-risk: 15.

По ROP validation pack v1:

- `P0` + no-live/artifact: 10;
- `Риск потери выручки` + no-live/artifact: 33;
- bot-candidate rows с деньгами/скидками/рассрочкой/возвратом/дедлайнами: 94.

## Что это меняет в плане правок

### 1. Нужен не regex-фильтр, а scoring/classifier

Минимальная логика:

- `strong_no_live_marker`;
- `asr_artifact_marker`;
- `system_no_dialogue_phrase`;
- `client_turn_length`;
- `client_live_business_terms`;
- `history_live_business_terms`;
- `structured_business_fields`;
- `duration/transcript_length`;
- `manager/client speaker balance`.

Итоговый статус:

- `non_conversation_high_confidence`;
- `manual_review_probable_no_live`;
- `manual_review_borderline_live_context`;
- `contentful_protected_live_dialogue`.

### 2. Backfill должен быть sidecar-first

Нельзя сразу перетирать `analysis_json` в SQLite.

Правильный порядок:

1. Сгенерировать sidecar CSV/JSONL по всем 64 832 звонкам.
2. Отдельно показать high-confidence, borderline и protected-live.
3. Проверить sample:
   - 50 high-confidence;
   - 50 borderline;
   - 50 protected-live.
4. Применять автоматически только high-confidence при precision >= 95%.
5. Borderline оставить на ручную проверку.
6. Старые поля не удалять: добавить `contentful_v2`, `call_type_v2`, `quality_reason_codes`.

### 3. Нужны защитные тесты на живые звонки

Regression dataset уже собран:

- `non_conversation_high_confidence`: 25;
- `manual_review_probable_no_live`: 15;
- `manual_review_borderline_live_context`: 10;
- `contentful_service_call_must_remain_live`: 10;
- `contentful_technical_call_must_remain_live`: 10;
- `contentful_sales_call_must_remain_live`: 10;
- `contentful_existing_client_progress_must_remain_live`: 10.

Этот датасет должен стать основой тестов перед изменением Analyze/readiness/pilot/KB/ROP.

## Примеры выводов

### High-confidence no-live

Пример:

`2025-11-12__16-01-26__Коршунова Анастасия__79067827796.mp3`

Summary:

`Абонент недоступен, звонок был перенаправлен на голосовую почту. Сообщение после сигнала оставить не удалось, содержательного диалога не было.`

Transcript:

`Зываемый абонент недоступен. Звонок был перенаправлен на голосовой почтовый ящик... Продолжение следует...`

Это должно быть `non_conversation`, не `technical_call/contentful`.

### False-positive guard: живой service call

Пример:

`2026-05-04__12-12-53__74997887317__Тютюнник Александр.mp3`

Есть слова `почта`, `перезвонить`, `письмо`, но это живой сервисный звонок: клиент обсуждает справку, заявление, чек и повторную отправку письма.

Такой звонок нельзя резать regex-ом по слову `почта`.

### False-positive guard: живой sales call

Пример:

`2026-05-05__18-12-56__79515838101__Тютюнник Александр.mp3`

Есть `перезвонить`, `сообщить`, сроки оплаты, но это живой sales call по летнему лагерю и оплате.

Такой звонок должен остаться contentful.

## Артефакты

- `stable_runtime/transcript_quality_adversarial_audit_20260509/summary.json`
- `stable_runtime/transcript_quality_adversarial_audit_20260509/ADVERSARIAL_AUDIT_REPORT.md`
- `stable_runtime/transcript_quality_adversarial_audit_20260509/regression_dataset.csv`
- `stable_runtime/transcript_quality_adversarial_audit_20260509/regression_dataset.jsonl`
- `stable_runtime/transcript_quality_adversarial_audit_20260509/high_confidence_no_live_or_artifact.csv`
- `stable_runtime/transcript_quality_adversarial_audit_20260509/borderline_probable_no_live.csv`
- `stable_runtime/transcript_quality_adversarial_audit_20260509/borderline_possible_false_positive.csv`
- `stable_runtime/transcript_quality_adversarial_audit_20260509/false_negative_marker_discovery_nonconv_old_missed.csv`
- `stable_runtime/transcript_quality_adversarial_audit_20260509/false_positive_guard_live_sales_call.csv`
- `stable_runtime/transcript_quality_adversarial_audit_20260509/false_positive_guard_live_service_call.csv`
- `stable_runtime/transcript_quality_adversarial_audit_20260509/false_positive_guard_live_technical_call.csv`
- `stable_runtime/transcript_quality_adversarial_audit_20260509/false_positive_guard_live_existing_client_progress.csv`

## Финальный вывод третьего аудита

Третий аудит подтвердил необходимость правок, но уточнил их безопасную форму.

Главный риск правок: если сделать слишком грубый фильтр, мы потеряем много живых звонков, потому что `почта`, `сообщение`, `перезвон`, `недоступен` встречаются и в нормальных продажах/сервисе.

Правильная стратегия:

1. Единый quality classifier.
2. High-confidence no-live auto-backfill только после dry-run.
3. Borderline только manual review.
4. Protected-live regression tests обязательны.
5. Bot safety sanitizer отдельно от no-live фильтра.
