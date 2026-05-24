# Current State

Дата обновления: 2026-05-23

Назначение: короткая актуальная точка правды по проекту. Если чат, старые документы и текущие файлы расходятся, сначала читать этот документ, затем `docs/DECISIONS_LOG.md`, `docs/ROADMAP.md`, `docs/RUNBOOK.md` и актуальное ТЗ.

## Короткий вывод

Проект находится на этапе внутреннего Telegram-пилота ИИ-сотрудника продаж. Распознавание и анализ звонков в целом доведены до сильного состояния; сейчас главный продуктовый риск не ASR, а управляемость Telegram-ботов: единый журнал пилота, объяснимость ответов, полезность диалога, контроль P0 и регулярный semantic review.

На 2026-05-23 основной runtime звонков обновлён после Mango-дозагрузки и принят как текущий слой:

- активный export: `stable_runtime/sales_master_export_20260523_audio_working_store_v1`;
- canonical DB: `stable_runtime/canonical_master_20260523_audio_working_store_v1/canonical_calls_master.db`;
- actionable звонков: `65 939`;
- missing ASR: `0`;
- missing Resolve+Analyze: `0`;
- AMO-ready после CRM quality gate: `2`;
- safe writeback pending: `0`.

Ключевое решение этого обновления: текущая точка правды фиксируется через `stable_runtime/CURRENT_RUNTIME.json` и `stable_runtime/CANONICAL_EXPORT.txt`; все актуальные аудиоссылки canonical DB переведены в единую рабочую папку `product_data/audio_working_store_20260523_v1/`.

Текущий правильный фокус:

1. Стабилизировать два публичных Telegram-бота: Фотон и УНПК МФТИ.
2. Закрыть `docs/TZ_TELEGRAM_PILOT_JOURNAL_AND_DIALOGUE_STRATEGY_2026-05-23.md`: единый журнал пилота, daily report, feedback import, очереди смысловой проверки и диалоговую стратегию нового лида.
3. Следующий слой качества: `docs/TZ_DIALOGUE_MEMORY_AND_FAILURE_SKILLS_2026-05-23.md` - диалоговая память, skill для классов ошибок, answer-first gate и честный holdout.
4. Прогнать быстрые тесты: `v8_targeted16`, затем статичные `v6/v5`; полный v8 запускать позже отдельным контролируемым прогоном.
5. Использовать историю Telegram/звонков/email как candidate-источник для KB/gold-текстов/тестов, но не добавлять факты в КБ автоматически.

## Текущая ветка и git-состояние

На момент последнего аудита:

- рабочая ветка: `codex/git-order-20260513`;
- последние зафиксированные блоки: TZ X/Y/Z;
- рабочая папка содержит много незакоммиченных изменений из параллельной разработки;
- Блок G зафиксирован в `docs/CURRENT_DEVELOPMENT_BOUNDARIES_2026-05-15.md`;
- Блок A зафиксирован коммитом `1cca43728`.
- Блок PBF зафиксирован коммитом `e49b81b93`.
- Блок B зафиксирован коммитом `91a3d694c`.
- Блок C зафиксирован коммитом `af444d561`.
- Блок D зафиксирован коммитом `135520a72`.
- Блок E реализован в текущем рабочем дереве и должен быть зафиксирован отдельным коммитом.

Не считать незакоммиченные файлы мусором без отдельного анализа.

## Главные уже сделанные вещи

- Построен pipeline распознавания и анализа звонков.
- Собраны цепочки общения по телефонам.
- После улучшения обработки звонков был пересобран post-backfill слой.
- Реализованы и приняты блоки X/Y/Z:
  - анализ звонков стал осторожнее;
  - sanitizer/quality стали надежнее;
  - часть hygiene-проверок закрыта.
- Создан deal-aware слой по сделкам AMO.
- Подготовлены батчи для РОП-проверки.
- Создан каталог клиентских вопросов и опросник для РОПа.
- Начаты Telegram/history/channel, mail archive и customer timeline направления.
- Установлены дополнительные Codex skills:
  - `security-best-practices`;
  - `security-threat-model`;
  - `security-ownership-map`;
  - `pdf`;
  - `jupyter-notebook`;
  - `cli-creator`.

## Главные текущие проблемы

1. Telegram-пилот должен стать измеримым: каждый ответ должен попадать в единый журнал с маршрутом, фактами, флагами, задержкой, контекстом и последующей оценкой сотрудника.
2. Бот уже безопаснее, чем в первых версиях, но ещё требует тюнинга пользы: прямой ответ на вопрос, память класса/предмета/формата, один следующий шаг и менее шаблонный тон.
3. Нельзя считать `quality_passed=true` готовностью клиентского слоя: для бота и базы знаний нужен отдельный `semantic_pass`.
4. Полный v8 ещё не является пройденным acceptance gate; сначала нужен `v8_targeted16` и быстрые `v6/v5`.
5. В `stable_runtime` и `product_data` ещё остаются крупные архивные слои; чистить их можно только через manifest и перенос в корзину.

## Текущий утвержденный порядок работ

Актуальное ТЗ:

`docs/TZ_TELEGRAM_PILOT_JOURNAL_AND_DIALOGUE_STRATEGY_2026-05-23.md`

Следующее ТЗ качества диалога:

`docs/TZ_DIALOGUE_MEMORY_AND_FAILURE_SKILLS_2026-05-23.md`

Порядок:

1. Не трогать работу параллельного диалога по audio working store и старым путям.
2. Закрыть единый Telegram pilot journal: SQLite store, daily report, feedback import, P0/reask/template/facts queues.
3. Довести dialogue strategy нового лида: прямой ответ, память контекста, один следующий шаг, тёплый тон, P0 без ослабления.
4. Обновить `CURRENT_STATE`, `DECISIONS_LOG`, `ROADMAP`, `RUNBOOK`.
5. Прогнать `v8_targeted16`.
6. После анализа targeted16 запускать статичные `v6/v5`; полный v8 - отдельным следующим блоком.

Исторический контур `G -> A -> PBF -> B -> C -> D -> E` по AMO/deal-aware/customer timeline считать фундаментом, но не текущим основным фокусом этого диалога.

## Runtime-истина

Для звонков и post-backfill слоя ориентироваться на:

- `stable_runtime/CURRENT_RUNTIME.json`
- `stable_runtime/CANONICAL_EXPORT.txt`
- `stable_runtime/canonical_master_20260523_audio_working_store_v1/summary.json`
- `stable_runtime/sales_master_export_20260523_audio_working_store_v1/summary.json`
- `stable_runtime/crm_writeback_quality_gate_20260523_audio_working_store_v1/summary.json`
- `stable_runtime/amo_writeback_queue_20260523_audio_working_store_v1/summary.json`

Единая рабочая папка аудиозаписей:

- `product_data/CURRENT_AUDIO_WORKING_STORE.txt`
- `product_data/audio_working_store_20260523_v1/`
- contract: `docs/AUDIO_WORKING_STORE_CONTRACT_2026-05-23.md`
- cleanup manifest: `docs/AUDIO_WORKING_STORE_CLEANUP_MANIFEST_2026-05-23.md`

Текущая canonical DB содержит `65974` ссылки на новый audio store и `0` ссылок на старые аудио-папки. Старые mp3-копии перенесены в корзину по manifest; служебные не-аудио файлы старых папок оставлены.

`stable_runtime` читать можно, но не менять без отдельного подтверждения.

## Что сейчас не делать

- Не запускать ASR.
- Не запускать R+A.
- Не писать в AMO/CRM/Tallanto.
- Не делать массовые batch-запуски.
- Не чистить `stable_runtime`.
- Не удалять архивы и старые батчи без отдельного подтверждения.
- Не делать новые параллельные крупные ветки разработки до завершения текущего ТЗ.

## Ближайший следующий шаг

Актуальный следующий шаг для Telegram-пилота:

- использовать `docs/TZ_TELEGRAM_PILOT_JOURNAL_AND_DIALOGUE_STRATEGY_2026-05-23.md`;
- для следующего слоя качества использовать `docs/TZ_DIALOGUE_MEMORY_AND_FAILURE_SKILLS_2026-05-23.md`;
- разбирать FAIL/PASS_WITH_NOTES через skill `bot-failure-class-review` и реестр `docs/BOT_FAILURE_CLASSES_REGISTRY.md`;
- строить дневной отчёт через `scripts/build_telegram_pilot_daily_report.py`;
- импортировать разметку сотрудников через `scripts/import_telegram_pilot_feedback.py`;
- проверять диалоговую стратегию через `scripts/run_telegram_dynamic_client_sim.py`;
- перед большим тестированием запускать `v8_targeted16`, а полный v8 оставлять на отдельный длинный прогон.

Предыдущий customer timeline шаг:

- создана локальная база `product_data/customer_timeline/deal_aware_sample_20260515/customer_timeline.sqlite`;
- повторный аудит на 100 группах дал покрытие 100/100;
- `ready_for_preview`: 18/100;
- `needs_manual_review`: 82/100;
- `timeline_preview_enabled` и `timeline_primary_read_enabled` не включаются.

Следующий крупный шаг:

- разобрать 82 ручных причины и разделить реальные проблемы истории от слишком строгих правил проверки;
- затем расширить локальный импорт на весь набор 709 сделок и добавить отдельную чистую контрольную выборку.
