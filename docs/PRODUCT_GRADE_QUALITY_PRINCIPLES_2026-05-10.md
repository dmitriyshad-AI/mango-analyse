# Product-Grade Quality Principles

Дата: 2026-05-10

## Цель

Mango analyse развивается не как локальный набор скриптов для одной компании, а как будущий SaaS / appliance-сервис для обработки звонков, CRM enrichment и sales intelligence.

Поэтому каждое исправление качества должно закрывать не один найденный пример, а устойчивый класс проблем, который может встретиться у другой организации, другого CRM-провайдера или другой телефонии.

## Текущий честный статус

По Stage15 transcript/bot-safety pipeline уже сделан сильный product-grade слой:

- есть frozen corpus;
- есть threat model;
- есть sanitizer/detector-подход;
- есть exit gates;
- есть Claude/GPT-аудит как внешний контроль;
- есть запрет на autonomous bot без отдельного gate.

По AMO post-backfill writeback слой `amo_post_backfill_writeback_20260510_v4_product_gate` уже переведен из literal-fix режима в class-based quality gate для staged AMO dry-run. Это все еще не финальная универсальная SaaS-архитектура, но ключевая ошибка v1-v3 - narrow-fix-then-regress - закрыта на уровне процесса.

Что уже сделано правильно:

- исправления переведены из literal-fix в class-based regex/heuristic layer;
- builder использует отдельный `crm_writeback_quality_detector`, а не локальный список частных regex;
- добавлен CRM writeback frozen corpus на 49 cases;
- добавлен full quality gate перед AMO-ready writeback;
- v1/v2 Claude findings превращены в регрессионные тесты;
- pointer переключен на post-backfill слой;
- protected AMO fields не пишутся;
- phone redaction добавлен как отдельный safety gate;
- low-value/out-of-domain/no-content rows удаляются из AMO-ready;
- full-scan counters добавлены в audit pack;
- строки с пустой историей или CRM-текстом, заканчивающимся `...`, блокируются для AMO-ready.

Что еще не полностью product-grade:

- low-value classifier пока является эвристическим regex/heuristic-слоем, а не полноценным tenant-configurable classifier;
- frozen corpus пока небольшой и покрывает текущую staged-writeback задачу, а не все будущие отрасли;
- нет tenant YAML/JSON-конфигурации для отрасли, продуктов, допустимых/недопустимых типов обращений;
- нет измеренной precision/recall на случайной выборке неотфильтрованных CRM-ready строк.

## Обязательный подход дальше

1. Не чинить только конкретные строки.

Каждая находка Claude/GPT/ручного аудита должна быть классифицирована как:

- новый пример уже известного класса;
- новый класс угрозы/шума;
- допустимый edge case;
- ложное срабатывание аудитора.

Код менять только после формулировки класса.

2. Каждый fix должен иметь regression test.

Минимум:

- synthetic examples для класса;
- реальные строки из аудита как fixture;
- negative examples, чтобы не выкинуть полезные продажи.

3. Builder и detector должны расходиться.

Для production-grade CRM writeback нужен отдельный detector:

- builder строит полезную историю;
- detector независимо проверяет, можно ли строку писать в CRM;
- detector не должен просто подтверждать собственные regex builder-а.

4. Sanity counters должны быть hard gates.

Если `crm_writeback_quality_blocking_rows > 0`, `phone_redaction_needed_rows > 0`, `ellipsis_rows > 0`, `empty_history_rows > 0` или protected-field violation > 0, pipeline должен завершаться ошибкой для live/staged writeback.

5. Конфигурация должна быть tenant-aware.

Для SaaS нельзя зашивать только наши бренды и наши частные паттерны.

Нужны конфиги:

- отрасль клиента;
- допустимые продукты;
- допустимые типы B2B-обращений;
- запрещенные out-of-domain категории;
- CRM fields mapping;
- protected fields;
- privacy/redaction policy.

6. Выходной критерий должен быть измеримым.

Перед live writeback:

- fixed audit corpus: 0 P0/P1;
- random sample: допустимый residual risk, явно записанный в отчете;
- staged dry-run: 0 protected-field violations, 0 API write errors, 0 obvious noise rows in sample;
- rollback path documented.

7. Claude/GPT-аудит используется как независимый контроль, не как бесконечная разработка.

Claude findings не принимаются автоматически.

Процесс:

- прочитать finding;
- классифицировать класс;
- согласиться или отклонить с причиной;
- изменить код/тесты;
- пересобрать artifacts;
- дать повторный audit pack.

## Практическая позиция по текущему AMO writeback

`amo_post_backfill_writeback_20260510_v4_product_gate` можно рассматривать как кандидат для real-tunnel AMO dry-run без live-записи после независимого Claude Code аудита.

Для live writeback по всем `5 667` строкам нужен один из двух вариантов:

1. Claude v4 дает PASS или PASS_WITH_LIMITATIONS без P1/P2 blockers.
2. Если Claude снова находит класс проблем, сначала переносим этот класс в detector/gate, а не делаем массовую запись.

## Следующий архитектурный шаг после AMO dry-run

Дальше развить `crm_writeback_quality_detector`:

- tenant-aware config;
- structured reasons: `out_of_domain`, `no_content`, `wrong_number`, `privacy_phone`, `protected_field`, `stale_context`, `manual_review_required`;
- CSV/XLSX report для РОПа;
- hard gate перед staged/live writeback.

Это переведет текущий качественный локальный слой в reusable SaaS-grade компонент.
