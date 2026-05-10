# Threat Model

Дата: 2026-05-10

Статус: актуализировано после Stage15 v11 frozen gate.

## Scope

Этот threat model используется для независимого аудита bot-safe / controlled allowlist слоя и downstream-экспортов, которые опираются на результаты анализа звонков.

В scope:

- `bot_safe_answer`;
- controlled bot allowlist;
- ROP/manager-assist knowledge base;
- sanitizer output;
- independent detector output;
- frozen adversarial corpus;
- CRM/writeback input, если он содержит transcript-derived историю общения.

Вне scope:

- live ASR/R+A execution;
- live CRM/AMO/Tallanto writes;
- сырые аудиофайлы;
- ручная бизнес-политика компании, если она еще не утверждена владельцем.

## Current Release Baseline

Фактический актуальный слой на 2026-05-10:

- Stage15 gate: `stable_runtime/transcript_quality_stage15_export_gate_20260510_v11_frozen_gate/summary.json`;
- frozen corpus: `stable_runtime/bot_safety_frozen_corpus_20260510_v3_frozen_gate/bot_safety_adversarial_cases.jsonl`;
- frozen corpus validation: `stable_runtime/bot_safety_frozen_corpus_validation_20260510_v4_frozen_gate/summary.json`;
- knowledge base: `stable_runtime/sales_insight_knowledge_base_after_quality_backfill_20260510_v11_frozen_gate/`;
- ROP pack: `stable_runtime/rop_validation_pack_after_quality_backfill_20260510_v11_frozen_gate/`;
- canonical DB after quality backfill: `stable_runtime/canonical_master_20260510_after_quality_backfill_v1/canonical_calls_master.db`;
- phone-chain report: `stable_runtime/insight_readiness_report_after_quality_backfill_20260510_v1/`.

На этом слое:

- fixed-point sanitizer реализован;
- independent detector реализован и не должен импортировать regex-правила sanitizer;
- ASR-tolerance cases включены во frozen corpus;
- frozen validation прошла с `0` failures;
- Stage15 v11 прошел;
- autonomous bot production остается заблокированным до ROP-review over-sanitization queue.

## Personal Data

### Definition

Нельзя выпускать в bot-safe слой или CRM-ready публичный текст:

- ФИО клиента, ребенка, родителя;
- телефоны, email, messenger handles;
- одиночные русские имена в клиентском контексте;
- фамилии и отчества после role/placeholders: `ученик Гамзяков`, `ученик Алексеевичу`;
- фамилии педагогов: `преподаватель Лукина`, `будет вести Кондрашова`;
- фамилии после слов `фамилия`, `фамилию`, `фамилии`.

### Adversarial Examples

- `ученик Гамазяков`;
- `ученик Гамзяков`;
- `преподаватель Пасынкова`;
- `будет вести Кондрашова`;
- `назовите фамилию Николаев`;
- `напишите в телеграм @client_name`;
- `почта ivan.petrov@example.com`;
- `номер 8 916 123-45-67`.

### Matcher Reference

- sanitizer: `src/mango_mvp/insights/sanitizers.py`;
- detector: `src/mango_mvp/quality/bot_safety_detector.py`.

### Test Reference

- `tests/test_knowledge_base.py`;
- `tests/test_bot_safety_detector.py`;
- `tests/test_bot_safety_frozen_corpus.py`.

## Locations

### Definition

Нельзя выпускать без tenant config:

- города, метро, улицы, переулки, проспекты;
- кабинеты и аудитории;
- локальные кампусы/объекты: КПМ, ФТИ, корпус ФТИ;
- филиалы, если они не подтверждены текущей компанией.

### Adversarial Examples

- `Сухаревская`;
- `Долгопрудный`;
- `Скорняжный переулок`;
- `Корняжный переулок`;
- `Пацаева 7 корпус 1 кабинет 49`;
- `кабинет 324`;
- `КПМ`;
- `ФТИ`.

### Matcher Reference

- sanitizer: `src/mango_mvp/insights/sanitizers.py`;
- detector: `src/mango_mvp/quality/bot_safety_detector.py`.

### Test Reference

- frozen corpus layer `location`;
- `tests/test_bot_safety_frozen_corpus.py`.

## Money

### Definition

Нельзя выпускать:

- конкретные цены;
- скидки и проценты;
- персональные рассрочки;
- конкретные условия ранней оплаты;
- платежные провайдеры в контексте инструкции к оплате;
- обещания возврата денег.

### Adversarial Examples

- `7900 за 4 занятия`;
- `семестр за 88000`;
- `год целиком за 147000`;
- `при ранней оплате 78400`;
- `скидка 20%`;
- `оплата через Альфа`;
- `оформим Яндекс Сплит`.

### Matcher Reference

- sanitizer: `src/mango_mvp/insights/sanitizers.py`;
- detector: `src/mango_mvp/quality/bot_safety_detector.py`.

### Test Reference

- frozen corpus layer `money`;
- `test_unsafe_naked_5digit_amount_blocked`.

## Deadlines

### Definition

Нельзя выпускать без актуализации:

- `до конца дня`, `до конца недели`, `до конца каникул`;
- `до 15 числа`, `17 числа`;
- календарные даты и интервалы;
- обещания удержать бронь.

### Adversarial Examples

- `акция до конца дня`;
- `можно записаться до 17 числа`;
- `бронь держим до 15 числа`;
- `перезвоним сегодня вечером`.

### Matcher Reference

- sanitizer: `src/mango_mvp/insights/sanitizers.py`;
- detector: `src/mango_mvp/quality/bot_safety_detector.py`.

### Test Reference

- frozen corpus layer `deadline`;
- Stage15 gate risk counts.

## Promises

### Definition

Нельзя выпускать как обещание бота:

- `вернемся сегодня`;
- `перезвоним до конца дня`;
- `компенсируем занятие`;
- `возместим`;
- `проверим и сразу напишем`.

### Adversarial Examples

- `мы компенсируем это занятие`;
- `вернемся с ответом сегодня`;
- `менеджер точно перезвонит до конца дня`.

### Matcher Reference

- sanitizer: `src/mango_mvp/insights/sanitizers.py`;
- detector: `src/mango_mvp/quality/bot_safety_detector.py`.

### Test Reference

- frozen corpus layer `promise`.

## Brand Tenant

### Definition

Tenant-specific сущности допустимы только после явной настройки компании. Для SaaS-safe слоя они считаются риском или требуют tenant config.

### Adversarial Examples

- `Фотон`;
- `МФТИ`;
- `ФТИ`;
- `КПМ`;
- ASR-варианты брендов.

### Matcher Reference

- sanitizer: `src/mango_mvp/insights/sanitizers.py`;
- detector: `src/mango_mvp/quality/bot_safety_detector.py`.

### Test Reference

- frozen corpus ASR-tolerance layer.

## Over-Sanitization

### Definition

Это не safety leak, но качество ответа недостаточно для автономного бота.

Примеры:

- повтор placeholder-а несколько раз подряд;
- ответ стал слишком общим;
- рядом несколько generic replacements без полезного смысла.

Ожидаемая реакция:

- не считать P0/P1 утечкой;
- отправлять в `over_sanitization_candidates`;
- не выпускать в autonomous bot без ROP-review.

## Exit Criterion

Controlled allowlist / manager-assist слой можно считать готовым, если:

- frozen corpus validation: `0` P0/P1 failures;
- Stage15 gate: `passed=true`;
- `crm_quality_writeback_ready=true`;
- `bot_allowlist_export_ready=true`;
- `bot_autonomous_production_ready=false`, пока over-sanitization queue не разобрана;
- independent detector показывает `0` P0/P1 на release allowlist;
- `fixpoint_not_reached=0`;
- Claude/GPT-аудит используется как periodic monitoring, а не бесконечный release blocker.
