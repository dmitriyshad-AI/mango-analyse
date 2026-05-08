# Доделки перед следующим этапом

Дата: 2026-05-07

Цель: зафиксировать конкретные хвосты обработки, которые нужно закрыть до пересборки contact-layer, ROP-пакетов, AMO refresh и следующей стадии product/SaaS-разработки.

## Текущий строгий статус покрытия

Источник отчета: `stable_runtime/final_processing_coverage_report_20260507_v3/summary.json`.

Период аудита: звонки из основной рабочей папки с января 2025 по май 2026 включительно.

Итого:

| Метрика | Количество |
|---|---:|
| Аудиофайлов в основной папке | 64 867 |
| С ASR | 64 703 |
| Без ASR | 164 |
| С полным Resolve+Analyze | 63 935 |
| Без полного Resolve+Analyze | 932 |
| Manual-хвосты без полного R+A | 468 |
| ASR есть, но R+A нет, не manual | 300 |

Вывод: основная история почти закрыта, но утверждать "обработано абсолютно все" пока нельзя.

## P0. Добить 164 звонка без ASR за март 2026

Файл со списком: `stable_runtime/final_processing_coverage_report_20260507_v3/missing_asr.txt`.

Статус 2026-05-07 13:55 MSK: actionable часть закрыта.

- `129` звонков с телефоном клиента обработаны ASR и R+A.
- DB: `stable_runtime/mar2026_client_asr_tail_129_20260507/mar2026_client_asr_tail_129_20260507.db`.
- R+A финальный отчет: `stable_runtime/resolve_analyze_mar2026_client_asr_tail_129_20260507/final_status.json`.
- Результат R+A: `resolve_done=78`, `resolve_skipped=51`, `resolve_manual=0`, `analysis_done=129`, `dead_letter=0`, `actionable=0`.
- `35` manager-manager звонков не распознавались по текущему решению; они явно вынесены в `stable_runtime/mar2026_client_asr_tail_129_20260507/exclusions/manager_manager_excluded.csv`.
- Coverage v4 учитывает эти 35 как `excluded_no_asr`, а не как actionable gap.

Что видно по первым примерам: часть файлов похожа на внутренние/manager-manager звонки, где вместо телефона в имени второй сотрудник. Это нужно не просто игнорировать, а явно классифицировать.

Действия:

1. Разделить 164 файла на клиентские и неклиентские.
2. Для клиентских сделать ASR.
3. Для неклиентских поставить явное исключение/статус `non_client_internal`, чтобы strict audit больше не считал их незавершенными.
4. После ASR сразу прогнать Resolve+Analyze там, где это применимо.
5. Пересобрать coverage v4 и проверить, что `missing_asr = 0` или все остатки имеют явное исключение.

Acceptance criteria:

- У каждого из 164 файлов есть один из статусов: `asr_done`, `non_client_internal`, `excluded_with_reason`.
- Нет молчаливого хвоста без статуса.
- Coverage report содержит отдельную строку по исключенным внутренним звонкам.

## P0. Добить 300 Jan 2025 ASR-only звонков из M1 test300

Суть: результаты ASR были перенесены из `external_m1_jan2025_test300`, но по строгому аудиту 300 январских звонков имеют ASR и не имеют полного R+A.

Статус 2026-05-07 13:43 MSK: закрыто.

- DB: `stable_runtime/external_m1_jan2025_test300_20260503/external_m1_jan2025_test300_20260503.db`.
- Запуск: `stable_runtime/start-external-m1-jan2025-test300-resolve-analyze-20260507.sh`.
- Финальный отчет: `stable_runtime/resolve_analyze_external_m1_jan2025_test300_20260507/final_status.json`.
- Результат: `resolve_done=230`, `resolve_skipped=70`, `resolve_manual=0`, `analysis_done=300`, `dead_letter=0`, `actionable=0`.
- Два manual-хвоста были финализированы как `skipped` с reason `manual_tail_finalized_as_short_low_content_dialogue_20260507`, потому что это короткие низкосодержательные диалоги.

Файл со списком: `stable_runtime/final_processing_coverage_report_20260507_v3/missing_full_ra.txt`.

Действия:

1. Проверить, что эти 300 записей действительно относятся к M1 test300 и не являются дублями более свежей обработки.
2. Собрать отдельный R+A batch только на эти 300.
3. Запустить Resolve в 2 потока, Analyze в 6 потоков.
4. Проверить, что Analyze не ждет Resolve и нет технических retry-хвостов.
5. Пересобрать coverage report.

Acceptance criteria:

- Все 300 имеют `resolve_status = done/skipped_with_reason` и `analysis_status = done/skipped_with_reason`.
- Если звонок неразговорный, это отражено в статусе и отчете.
- В месяце 2025-01 остаток `asr_no_full_ra_non_manual` равен 0.

## P0. Разобрать 468 manual Resolve tails

Файл со списком: `stable_runtime/final_processing_coverage_report_20260507_v3/manual_not_full_ra.txt`.

Суть: это не массовая рабочая очередь Analyze, а хвосты, которые текущий Resolve считает ручными/проблемными. Их нельзя бесконечно оставлять в неопределенном состоянии, потому что они мешают честному статусу "история закрыта".

Действия:

1. Разделить 468 звонков на группы причин: короткий/пустой звонок, manager-manager, плохое аудио, конфликт каналов, техническая ошибка, реально сложный диалог.
2. Для очевидных non-conversation поставить финальный статус `skipped_non_conversation` с reason.
3. Для технических ошибок сделать controlled retry с меньшим параллелизмом.
4. Для реально сложных диалогов сделать rescue Resolve или ручную разметку ограниченной выборки.
5. После финализации пересобрать coverage report.

Acceptance criteria:

- Ни один звонок не остается просто `manual/pending` без финального решения.
- У каждого звонка есть либо полный R+A, либо финальный skip/manual reason.
- Coverage report отдельно считает `full_ra`, `final_skipped`, `needs_human_review`.

## P1. После закрытия P0 пересобрать рабочие артефакты

Действия:

Статус 2026-05-07 13:55 MSK: coverage v4 пересобран.

- Report: `stable_runtime/final_processing_coverage_report_20260507_v4/`.
- `source_audio=64867`.
- `excluded_no_asr=35`.
- `actionable_source_audio=64832`.
- `asr_done=64832`.
- `missing_asr_actionable=0`.
- `full_ra=64364`.
- `missing_full_ra_actionable=468`.
- `manual_not_full_ra=468`.
- `asr_no_full_ra_non_manual=0`.
- `errors=[]`.

1. Пересобрать canonical calls/contact-layer на максимально полной истории.
2. Проверить, что длинные истории и конспекты не обрезаются в `.xlsx`.
3. Пересобрать ROP-пакет: Reopen, Follow-up, Manual review, Top priorities, инструкции.
4. Пересобрать AMO-ready diff: новые записи и refresh существующих.
5. Перед AMO writeback сделать dry-run и quality gate.

Acceptance criteria:

- Новый coverage report сохранен в `stable_runtime/final_processing_coverage_report_YYYYMMDD_v4/`.
- В ROP/AMO таблицах нет обрезания содержательных историй.
- Все кандидаты в AMO имеют `writeback_allowed` или понятный blocker.

## P2. Правило на будущее

Перед каждым заявлением "вся история обработана" использовать только строгий coverage report, а не статус последнего batch. Batch может быть завершен, но общий корпус еще может иметь хвосты из других источников, manual-статусы или исключения без reason.
