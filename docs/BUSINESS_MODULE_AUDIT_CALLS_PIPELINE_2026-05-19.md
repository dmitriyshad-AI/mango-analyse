# Business Module Audit: Calls Pipeline

Дата: 2026-05-19

Источник задания: `docs/BUSINESS_AUDIT_PROMPTS_2026-05-19/02_CALLS_PIPELINE_AUDIT_PROMPT.md`

Режим проверки: read-only. ASR, Resolve+Analyze, live-записи в AMO/CRM/Tallanto и изменения `stable_runtime` не запускались.

## 1. Краткий вердикт

**Вердикт: PASS_WITH_NOTES для внутреннего manager-assist использования.**

Слой звонков уже полезен для отдела продаж: есть актуальный принятый canonical master, цепочки по телефонам, классификация содержательных/несодержательных звонков, выгрузка для AMO и набор защит от плохих CRM-текстов.

**Но это не `semantic_pass` для автономного клиентского бота и не полная готовность к массовой записи в CRM.** Зеленые технические проверки подтверждают структуру и формальные ограничения, но не доказывают, что смысл каждого резюме и следующего шага корректен для клиента/сделки.

Главный вывод: pipeline технически зрелый, но сейчас находится между двумя состояниями:

- принятый production-runtime: слой от 2026-05-16;
- более свежие звонки от 2026-05-17 уже частично/почти полностью обработаны в рабочей DB, но еще не приняты в основной runtime и downstream-слои.

## 2. Что проверялось

Проверены документы, runtime-указатели, audit packs, summary-файлы, код pipeline и выборочные SQLite-состояния.

Основные источники:

- `docs/CURRENT_STATE.md`
- `docs/DECISIONS_LOG.md`
- `docs/ROADMAP.md`
- `docs/RUNBOOK.md`
- `docs/AUDIO_STORE_RUNTIME_CONTRACT_2026-05-16.md`
- `docs/CANONICAL_MASTER_BUILD_REPORT_2026-05-09.md`
- `docs/RESOLVE_LLM_DISABLED_2026-05-15.md`
- `docs/CUSTOMER_TIMELINE_INTEGRATION_DECISION_2026-05-15.md`
- `docs/DEAL_AWARE_STAGE709_REVIEW_2026-05-14.md`
- `docs/TELEGRAM_PILOT_AND_CONTEXTUAL_CATALOG_STRATEGY_2026-05-17.md`
- `stable_runtime/CURRENT_RUNTIME.json`
- `stable_runtime/canonical_master_20260516_after_mango_update_v1/summary.json`
- `stable_runtime/insight_readiness_report_after_mango_update_20260516_v1/summary.json`
- `stable_runtime/sales_master_export_20260516_after_mango_update_v1/summary.json`
- `stable_runtime/crm_writeback_quality_gate_20260516_after_mango_update_v1/summary.json`
- `stable_runtime/amo_writeback_queue_20260516_after_mango_update_v1/summary.json`
- `product_data/mango_missing_1372_asr_only_20260517_v1/integration_summary_20260517_v1.json`
- `stable_runtime/ra_pending_mango_api_20260517_v1/ra_pending_mango_api_20260517_v1.sqlite`

Проверенные кодовые зоны:

- `src/mango_mvp/services/worker.py`
- `src/mango_mvp/services/transcribe.py`
- `src/mango_mvp/services/resolve.py`
- `src/mango_mvp/services/analyze.py`
- `src/mango_mvp/maintenance/canonical_master.py`
- `scripts/build_canonical_after_mango_update.py`
- `scripts/build_insight_readiness_from_canonical.py`
- `scripts/build_post_backfill_amo_ready_export.py`
- `src/mango_mvp/productization/call_processing_readiness.py`
- `src/mango_mvp/audio_store.py`
- `scripts/build_audio_store_downstream_projection.py`
- `src/mango_mvp/quality/tenant_text_normalizer.py`
- `src/mango_mvp/quality/non_conversation.py`

## 3. Текущий источник истины

Текущий принятый runtime зафиксирован в `stable_runtime/CURRENT_RUNTIME.json`.

Ключевые указатели:

- canonical DB: `stable_runtime/canonical_master_20260516_after_mango_update_v1/canonical_calls_master.db`
- active export root: `stable_runtime/sales_master_export_20260516_after_mango_update_v1`
- active export pointer: `stable_runtime/CANONICAL_EXPORT.txt` -> `sales_master_export_20260516_after_mango_update_v1`
- insight report: `stable_runtime/insight_readiness_report_after_mango_update_20260516_v1`
- CRM gate: `stable_runtime/crm_writeback_quality_gate_20260516_after_mango_update_v1`
- AMO queue: `stable_runtime/amo_writeback_queue_20260516_after_mango_update_v1`

Текущее принятое состояние:

| Показатель | Значение |
|---|---:|
| canonical actionable calls | 65 100 |
| missing ASR | 0 |
| missing Resolve+Analyze | 0 |
| AMO-ready rows | 6 |
| safe writeback pending rows | 0 |
| runtime validation checks | 16/16 ok |

Это означает: принятый слой от 2026-05-16 формально полный по ASR и R+A для своих 65 100 actionable звонков.

## 4. Как сейчас устроен путь звонка

Упрощенный путь:

1. Запись приходит из Mango.
2. Аудио кладется в canonical audio store / рабочие audio-папки.
3. ASR делает расшифровку.
4. Resolve собирает финальный текст разговора из доступных распознаваний и технических признаков.
5. Analyze извлекает смысл: тип звонка, продукты, возражения, следующий шаг, summary.
6. Canonical master собирает единую таблицу звонков.
7. Phone-chain собирает историю по нормализованному телефону.
8. Sales export / AMO-ready готовит управленческий слой для CRM.
9. Quality gates блокируют опасные или слабые строки.
10. AMO/writeback пишет только разрешенные AI-поля и только после dry-run/readback.

Важный технический факт: команда `run-all` в CLI по умолчанию не делает live sync во внешние системы. Запись в AMO вынесена в отдельные fail-closed скрипты.

## 5. Что уже реально хорошо

### 5.1. Принятый runtime не указывает на старый апрельский слой

`CANONICAL_EXPORT.txt` сейчас указывает на слой 2026-05-16, а не на старую апрельскую выгрузку. Это устраняет один из главных старых рисков: случайно писать в CRM устаревшую историю.

### 5.2. ASR/R+A для принятого слоя закрыты формально

Для canonical master 2026-05-16:

- `missing_asr=0`
- `missing_ra=0`

То есть внутри принятого слоя нет дырок вида “звонок есть, но расшифровки или анализа нет”.

### 5.3. Есть разделение содержательных и несодержательных звонков

В активном sales export:

| Тип | Кол-во |
|---|---:|
| sales_call | 35 797 |
| service_call | 7 665 |
| existing_client_progress | 2 140 |
| non_conversation | 18 847 |
| technical_call | 651 |

Это важно: pipeline уже не воспринимает все звонки как лиды и умеет отсеивать IVR, ошибочные, технические и нерелевантные звонки.

### 5.4. Есть phone-chain слой

`client_chains.csv` содержит 16 002 уникальных нормализованных телефона.

`calls_terminal_analyzed.csv` содержит 64 047 звонков, из них:

- contentful: 42 674;
- non-contentful: 21 373.

Это уже основа для истории общения по телефону, но еще не полноценная история по человеку/семье/сделке.

### 5.5. Есть независимые защитные слои

Сейчас есть отдельные проверки для:

- несодержательных звонков;
- CRM-текста;
- stale next step;
- ellipsis/truncation;
- duplicate label/count artifacts;
- protected AMO fields;
- phone redaction;
- deal-aware selection;
- customer timeline read-only integration.

Это правильное направление: проблема больше не решается только точечными grep-патчами.

### 5.6. Resolve LLM отключен по умолчанию

Это правильно. Resolve не должен “додумывать” разговор. Он должен аккуратно собирать аудио-основанный текст. LLM-улучшение допустимо только выборочно и только как безопасный выбор между audio-derived вариантами.

## 6. Что уже сделано, но пока не полностью используется

### 6.1. X-v3 Analyze улучшения внедрены в код, но не массово применены к истории

По активной canonical DB распределение версий примерно такое:

| analyze_prompt_version | Кол-во |
|---|---:|
| v6 | 52 025 |
| v7 | 4 |
| NULL | 13 071 |

И по профилю:

| analyze_prompt_profile | Кол-во |
|---|---:|
| compact | 38 002 |
| full | 14 027 |
| NULL | 13 071 |

Вывод: улучшения Analyze v7/X-v3 есть в коде, но исторические 65k звонков в основном не пересобраны этим новым качеством.

Это не значит, что всё плохо. Но нельзя утверждать, что вся историческая база уже получила последние улучшения анализа.

### 6.2. 2026-05-17 слой обработан дальше, чем говорит stale summary

Файл `product_data/mango_missing_1372_asr_only_20260517_v1/integration_summary_20260517_v1.json` говорит про ASR-only слой и `missing_full_ra=1372`.

Но фактическая SQLite DB `stable_runtime/ra_pending_mango_api_20260517_v1/ra_pending_mango_api_20260517_v1.sqlite` показывает более свежее состояние:

| Стадия | Статус |
|---|---:|
| transcription done | 1 372 / 1 372 |
| resolve done | 898 |
| resolve skipped | 457 |
| resolve manual | 17 |
| analysis done | 1 355 |
| analysis pending | 17 |

Вывод: R+A по свежим 1 372 звонкам почти завершен, но эти результаты еще не приняты в основной runtime и downstream-слои. Осталось 17 manual/pending строк.

Это важный разрыв между “рабочей DB уже продвинулась” и “официальный runtime еще не обновлен”.

### 6.3. Tenant normalizer уже есть

Есть `src/mango_mvp/quality/tenant_text_normalizer.py`, который закрывает часть известных текстовых ошибок:

- `МПК/НПК/... МФТИ` -> `УНПК МФТИ`;
- `летние ночные школы` -> `летние очные школы`;
- часть product/count artifacts.

Но нужно убедиться, что этот normalizer применяется во всех downstream-ветках, где текст попадает в CRM, deal-aware, Telegram и базы знаний.

## 7. Старые/опасные/неактуальные артефакты

### 7.1. Неактивные stable_runtime слои нельзя использовать как источник истины

В `stable_runtime` много исторических слоев. Это нормально для разработки, но опасно для ручной навигации.

Правило: источником истины является только `stable_runtime/CURRENT_RUNTIME.json` и указанные в нем папки.

### 7.2. 2026-05-17 слой нельзя использовать для CRM/bot до принятия

Несмотря на то, что R+A почти готов, слой 2026-05-17 еще не интегрирован в current runtime. Его нельзя подмешивать в CRM, Telegram или deal-aware без отдельного rebuild + gates.

### 7.3. Audio store cleanup candidates нельзя удалять сейчас

`docs/AUDIO_STORE_RUNTIME_CONTRACT_2026-05-16.md` прямо фиксирует, что duplicate cleanup candidates не являются разрешением на удаление.

Без отдельного cleanup-плана ничего не удалять.

### 7.4. Active export summary ссылается на XLSX, которых нет рядом

В `stable_runtime/sales_master_export_20260516_after_mango_update_v1/summary.json` указаны XLSX-артефакты, но в папке фактически лежат CSV и summary.

Отсутствуют:

- `master_contacts_ru.xlsx`
- `amo_export_ready_ru.xlsx`
- `master_export_pack_ru.xlsx`

Это не ломает CSV-пайплайн, но ломает ожидание менеджерской поставки в Excel.

## 8. Основные проблемы по приоритетам

### P1. Свежие 1 372 звонка не приняты в основной runtime

Рабочая DB показывает, что R+A почти завершен: 1 355 analysis done, 17 pending/manual.

Но текущий production-runtime всё еще 2026-05-16 и не включает этот слой как accepted downstream.

Риск: менеджеры/бот/CRM видят не самую свежую историю клиента.

Что нужно сделать:

1. Разобрать 17 `resolve_manual` / `analysis_pending` строк.
2. Собрать новый canonical master после 2026-05-17 R+A.
3. Пересобрать phone-chain, sales export, deal-aware, AMO-ready, Telegram/context layers.
4. Прогнать quality gates.
5. Обновить `CURRENT_RUNTIME.json` только после green gates и audit pack.

### P1. Нет полноценного смыслового golden corpus для звонков

Есть много формальных тестов и gate-ов, но нет достаточного набора “сырой разговор -> ожидаемый смысл -> ожидаемые CRM/deal-aware поля”.

Риск: pipeline формально зеленый, но смысл может быть слабым: неверный следующий шаг, устаревшее возражение, неверный продукт, смешение сделок, плохой статус оплаты.

Что нужно сделать:

- собрать 100-200 эталонных звонков с ручной разметкой РОП/Дмитрия;
- включить разные классы: оплата, отказ, дубль, действующий клиент, сервисный звонок, нецелевой звонок, перенос, договор, лагерь, ЛОШ, УНПК, занятия;
- сделать regression gate не только по regex, но и по ожидаемым бизнес-полям.

### P1. Новые улучшения Analyze не применены к исторической базе

X-v3/v7 улучшения есть, но в активной базе почти все звонки обработаны v6/compact или имеют NULL по версии.

Риск: downstream строится на старых summary, где возможны шумные продукты, слабые возражения, плохие next step, устаревшие формулировки.

Решение: не делать полный re-analyze 65k сразу. Нужен selective re-analyze:

- активные сделки;
- high-value clients;
- строки с quality flags;
- последние содержательные звонки;
- клиенты, попадающие в Telegram/AMO/deal-aware pilot.

### P1. Autonomous bot на базе звонков пока блокировать

Для manager-assist звонки уже полезны. Для автономного клиентского бота слой пока недостаточен.

Причины:

- нет semantic_pass на большом golden corpus;
- 2026-05-17 слой не принят;
- часть исторических звонков обработана старым Analyze;
- Customer Timeline пока не primary read;
- есть риск устаревших next steps и смешения контекста по телефону/сделке.

### P2. Customer Timeline пока не основной источник истории

По решению `CUSTOMER_TIMELINE_INTEGRATION_DECISION_2026-05-15.md`, timeline должен быть read-only источником. Но он еще не должен включаться как primary read.

Состояние из CURRENT_STATE: sample 100 groups, coverage 100/100, ready_for_preview 18/100, needs_manual_review 82/100.

Риск: если включить слишком рано, можно смешать события, сделки, телефоны и учеников.

### P2. Не все downstream ветки гарантированно используют tenant normalizer

Нормализатор брендов и известных ASR-ошибок есть, но нужно проверить его применение во всех финальных текстовых ветках.

Особенно важно для:

- AMO contact fields;
- AMO deal fields;
- Telegram draft context;
- question catalog;
- knowledge base;
- manager preview.

### P2. Active export содержит только 6 AMO-ready строк

Это не баг само по себе: после строгих фильтров массовая запись в AMO была ограничена.

Но бизнес-ожидание может быть другим. Если цель — большая CRM-актуализация, нужно расширять eligible population через deal-aware и manual-resolution, а не через ослабление safety gates.

### P2. Summary active export ссылается на отсутствующие XLSX

Для технических CSV это не критично. Для передачи РОПу/менеджерам это проблема упаковки.

Нужно либо восстановить XLSX-генерацию, либо убрать ссылки из summary.

### P3. Много исторических runtime-артефактов

Нужно навести порядок через quarantine/manifest, но не удалять сейчас без отдельного подтверждения.

### P3. Audio store v1 нужно явно сверить после 2026-05-17

Audio store contract закрывает 2026-05-16 слой и 267 новых Mango recordings. После скачивания/ASR новых звонков нужно сделать v2 projection или отдельный accepted audio inventory.

## 9. Тесты: что есть и чего не хватает

### Уже есть сильные формальные тесты

Покрыты зоны:

- analyze form/guards;
- resolve form/fallback;
- canonical master provenance;
- non-conversation detector;
- CRM text quality detector;
- CRM writeback frozen corpus;
- post-backfill AMO-ready export;
- productization call readiness;
- audio store projection;
- deal-aware attribution;
- customer timeline provider.

Примеры файлов:

- `tests/test_analyze.py`
- `tests/test_analyze_xa_safe_pack.py`
- `tests/test_non_conversation_quality.py`
- `tests/test_resolve.py`
- `tests/test_canonical_master.py`
- `tests/test_post_backfill_amo_ready_export.py`
- `tests/test_crm_text_quality_detector.py`
- `tests/test_crm_writeback_quality_detector.py`
- `tests/test_crm_writeback_frozen_corpus.py`
- `tests/test_productization_call_processing_readiness.py`
- `tests/test_audio_store_projection.py`
- `tests/test_deal_aware_stage2_attribution.py`
- `tests/test_customer_timeline_context_provider.py`

### Главная дыра

Нет достаточного смыслового end-to-end набора:

`raw/ASR transcript -> Resolve -> Analyze -> canonical -> phone-chain -> deal-aware/CRM text -> expected business meaning`

Нужны тесты не только “поле заполнено”, но и “вывод верен по смыслу”.

## 10. Достаточен ли слой звонков для Telegram bot / CRM

### Telegram manager-draft pilot

**Можно использовать с ограничениями.**

Условия:

- только черновики для менеджера;
- без автоматической отправки клиенту;
- показывать источник/фрагмент звонка;
- блокировать ответы при weak context: нет локального фрагмента, speaker_uncertain, old_summary_only, конфликт AMO/Tallanto, stale next step;
- semantic review обязателен.

### Telegram autonomous bot

**Нельзя включать сейчас.**

Причины:

- не принят свежий 2026-05-17 слой;
- нет достаточного semantic golden corpus;
- Customer Timeline еще не primary;
- old Analyze покрывает большую часть истории;
- риск устаревших коммерческих советов.

### CRM/AMO manager-assist

**Можно использовать staged, как уже делали: маленькими партиями, dry-run -> audit -> live -> readback.**

Нельзя делать широкий writeback без:

- deal-aware consistency;
- readback gate;
- protected field guard;
- semantic sample review;
- snapshot/rollback contract.

### Deal-aware CRM fields

Следующий правильный слой — deal-aware, а не contact-only. РОП уже подтвердил, что менеджеры работают в сделках. Контактная карточка полезна, но недостаточна.

## 11. Нужно ли сейчас переобрабатывать все звонки

**Нет, полный Super Resolve / полный re-analyze всех 65k сейчас не рекомендован.**

Причины:

- дорого;
- долго;
- есть риск, что LLM начнет “улучшать” смысл вместо сохранения аудио-фактов;
- большая часть бизнес-эффекта достигается выборочной переобработкой активных/рискованных/свежих звонков.

Рекомендуемый подход:

1. Сначала принять 2026-05-17 слой после 17 manual rows.
2. Запустить deterministic normalizer на закрытых классах ошибок.
3. Сделать выборочный re-analyze для активных сделок и проблемных строк.
4. Добавить semantic golden corpus.
5. Только после метрик решать, нужен ли более широкий re-analyze.

## 12. Что строго не трогать

Без отдельного явного подтверждения Дмитрия не делать:

- удаление старых audio/runtime/transcript файлов;
- массовый перенос stable_runtime артефактов;
- полный Super Resolve всех звонков;
- полный re-analyze 65k;
- live writeback в AMO/CRM;
- запись в Tallanto;
- включение Telegram autonomous replies;
- переключение Customer Timeline в primary read;
- использование 2026-05-17 слоя как current runtime без rebuild/gates.

## 13. Рекомендуемый следующий план

### Шаг 1. Закрыть 2026-05-17 слой

- Найти 17 `resolve_manual` / `analysis_pending` строк.
- Решить: ручной разбор, quarantine или safe skip.
- Зафиксировать причину по каждой строке.

### Шаг 2. Собрать новый accepted runtime

После закрытия 17 строк:

- canonical master;
- phone-chain;
- sales export;
- CRM quality gate;
- AMO queue;
- deal-aware projection;
- Telegram/context projection, если требуется.

### Шаг 3. Проверить tenant normalizer по всем финальным веткам

Цель: исключить повторение ошибок `МПК/НПК МФТИ`, `летние ночные школы`, product/count artifacts.

Нужно сделать один audit pack:

- где normalizer применяется;
- где не применяется;
- какие строки изменяются;
- есть ли over-normalization.

### Шаг 4. Создать semantic golden corpus для звонков

Минимум 100 строк:

- 50 активные сделки;
- 20 спорные оплаты/договоры;
- 10 сервисные звонки;
- 10 нецелевые/ошибочные;
- 10 действующие клиенты/дубли/старые сделки.

Для каждой строки нужны ожидаемые поля:

- call_type;
- продукт;
- актуальный статус;
- следующий шаг;
- возражения;
- payment/deal consistency;
- можно ли писать в CRM;
- можно ли использовать в Telegram draft.

### Шаг 5. Selective re-analyze

Не весь corpus, а только:

- последние звонки активных сделок;
- звонки с prompt_version NULL/v6 и высоким бизнес-весом;
- строки с stale next step/conflict/payment risk;
- строки, которые попадают в РОП/Telegram/AMO previews.

### Шаг 6. Deal-aware first

Перед новым массовым contact writeback двигаться в сторону deal-aware:

- писать краткую историю в сделку;
- учитывать статус сделки, оплату, задачи, последние события;
- не дублировать текст между полями;
- делать compact summary для менеджера и full history как отдельный слой.

### Шаг 7. Customer Timeline preview

Включать только как read-only preview:

- сначала sample;
- потом manager review;
- затем quality gate;
- только после этого primary read.

### Шаг 8. Навести порядок в runtime-артефактах

Не удалять. Сначала:

- manifest всех active/legacy слоев;
- quarantine candidates;
- why-keep/why-archive;
- отдельное подтверждение Дмитрия.

## 14. Ответы на ключевые вопросы из prompt

### Можно ли считать слой звонков достаточным для пилота?

Для внутреннего manager-assist пилота — да, с ограничениями и semantic review.

Для автономного клиентского бота — нет.

Для широкого CRM writeback — только staged: dry-run, audit, live small batch, readback.

### Какие 3-5 улучшений дадут максимум пользы?

1. Принять 2026-05-17 R+A слой в current runtime после закрытия 17 manual rows.
2. Создать semantic golden corpus по звонкам и бизнес-полям.
3. Проверить применение tenant normalizer во всех финальных текстовых ветках.
4. Сделать selective re-analyze активных/рискованных звонков новым Analyze.
5. Перейти от contact-only к deal-aware summaries для сделок.

### Нужно ли сейчас делать reprocess?

Да, но выборочный.

Полный reprocess 65k сейчас не нужен. Сначала закрыть свежие 1 372, затем re-analyze только активные, рискованные и бизнес-важные звонки.

### Что строго не трогать?

- Не удалять audio/runtime/transcript артефакты.
- Не включать 2026-05-17 слой как current без rebuild/gates.
- Не запускать полный Super Resolve.
- Не делать массовый live writeback.
- Не включать autonomous bot.
- Не использовать Customer Timeline как primary read.

## 15. Вопросы к Дмитрию

1. По 17 manual rows из 2026-05-17: предпочитаем ручной разбор, quarantine или исключение из accepted runtime на первом проходе?
2. Для semantic golden corpus: кто будет финальным арбитром по смыслу — Дмитрий, РОП или оба?
3. Для следующего пилота важнее сначала deal-aware AMO fields или Telegram manager-draft на базе звонков?
4. Нужно ли восстановить XLSX-выгрузки для РОП прямо сейчас, или CSV достаточно до следующего менеджерского ревью?

## 16. Итог

Проект по звонкам продвинулся далеко: есть формально полный accepted runtime, нормальная структура pipeline, защитные gate-ы и понятный путь в CRM/Telegram/deal-aware.

Главная текущая проблема не в отсутствии pipeline, а в разрыве между технической готовностью и смысловой надежностью. Следующий большой этап должен закрыть этот разрыв: принять свежие звонки, добавить смысловой эталон, выборочно переобработать важные звонки и перейти к deal-aware использованию в сделках.
