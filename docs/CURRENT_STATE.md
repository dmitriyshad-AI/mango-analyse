# Current State

Дата обновления: 2026-05-16

Назначение: короткая актуальная точка правды по проекту. Если чат, старые документы и текущие файлы расходятся, сначала читать этот документ, затем `docs/DECISIONS_LOG.md`, `docs/ROADMAP.md`, `docs/RUNBOOK.md` и актуальное ТЗ.

## Короткий вывод

Проект находится между исследовательской стадией и первым внутренним рабочим запуском. Распознавание и анализ звонков в целом доведены до сильного состояния. Главный риск сейчас не ASR, а безопасная запись результата в AMO, единая история клиента, порядок в git и управляемость runtime-артефактов.

На 2026-05-16 основной runtime звонков обновлён после Mango-дозагрузки:

- активный export: `stable_runtime/sales_master_export_20260516_after_mango_update_v1`;
- canonical DB: `stable_runtime/canonical_master_20260516_after_mango_update_v1/canonical_calls_master.db`;
- actionable звонков: `65 100`;
- missing ASR: `0`;
- missing Resolve+Analyze: `0`;
- телефонов в phone-chain слое: `16 002`;
- AMO-ready после CRM quality gate: `6`;
- safe writeback pending: `0`.

Ключевое решение этого обновления: старые и новые звонки собраны в новую версионированную runtime-базу, но live-запись в AMO/Tallanto не выполнялась.

Текущий правильный фокус:

1. Закрыть качество понимания клиентских вопросов для Telegram-пилота.
2. Сделать безопасную LLM-калибровку question catalog по 9 969 вопросам.
3. После этого возвращаться к customer timeline и следующему rebuild-раунду.

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

1. Качество понимания темы вопроса остается главным блокером Telegram-пилота: rule-only baseline на 100 размеченных строках дает около 37% accuracy и macro-F1 около 0.326.
2. Старый Codex A/B прогон лучше rule-only, но тоже ниже целевого порога 0.85; полный прогон 9 969 вопросов нельзя считать боевым без нового безопасного контура и ручной проверки спорных строк.
3. Customer timeline получил read-only adapter и контрольные выборки, но флаги `timeline_preview_enabled` и `timeline_primary_read_enabled` пока не включаются.
4. В `stable_runtime` много версий deal-aware артефактов без единой карты актуальности.

## Текущий утвержденный порядок работ

Основное ТЗ:

`docs/TOP3_PRIORITY_FIXES_TZ_2026-05-15.md`

Порядок:

1. `G` - git-границы и рабочее состояние. Статус: выполнено.
2. `A` - AMO pre-write snapshot и rollback. Статус: выполнено в коде, без live-запуска.
3. `PBF` - красный post-backfill тест. Статус: выполнено в коде.
4. `B` - коммерческие поля. Статус: выполнено в коде.
5. `C` - структурные возражения. Статус: выполнено в коде.
6. `D` - связь каталога вопросов и deal-aware gate. Статус: выполнено в коде и закоммичено.
7. `E` - customer timeline как read-only источник истории клиента. Статус: выполнено в коде, без live-запуска.

## Runtime-истина

Для звонков и post-backfill слоя ориентироваться на:

- `stable_runtime/CURRENT_RUNTIME.json`
- `stable_runtime/CANONICAL_EXPORT.txt`
- `stable_runtime/canonical_master_20260516_after_mango_update_v1/summary.json`
- `stable_runtime/insight_readiness_report_after_mango_update_20260516_v1/summary.json`
- `stable_runtime/sales_master_export_20260516_after_mango_update_v1/summary.json`

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

- использовать `docs/QUESTION_CATALOG_LLM_CALIBRATION_TZ_2026-05-16.md`;
- прогонять full-run только через `scripts/run_question_catalog_codex_full_v2.py`;
- писать результаты полного LLM-прогона только в `.codex_local/question_catalog/codex_full_v2/<run_id>/`;
- пересобирать каталог только в отдельную папку через `scripts/rebuild_question_catalog_from_llm_predictions_v2.py`;
- не перезаписывать текущий `product_data/question_catalog` до отдельного решения.

Предыдущий customer timeline шаг:

- создана локальная база `product_data/customer_timeline/deal_aware_sample_20260515/customer_timeline.sqlite`;
- повторный аудит на 100 группах дал покрытие 100/100;
- `ready_for_preview`: 18/100;
- `needs_manual_review`: 82/100;
- `timeline_preview_enabled` и `timeline_primary_read_enabled` не включаются.

Следующий крупный шаг:

- разобрать 82 ручных причины и разделить реальные проблемы истории от слишком строгих правил проверки;
- затем расширить локальный импорт на весь набор 709 сделок и добавить отдельную чистую контрольную выборку.
