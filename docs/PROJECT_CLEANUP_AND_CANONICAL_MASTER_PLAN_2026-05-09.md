# Project Cleanup And Canonical Master Plan

Дата: 2026-05-09

## Контекст

Проект перешел из режима постоянной добивки ASR/R+A в режим качества, консолидации и подготовки к следующей стадии: SaaS/productization, knowledge-base, CRM writeback и стабильная эксплуатация.

По последнему coverage v5 рабочий клиентский корпус за период `2025-01-01` - `2026-05-31` технически полностью обработан.

Источник проверки:

- `stable_runtime/final_processing_coverage_report_20260507_v5/summary.json`
- `stable_runtime/final_processing_coverage_report_20260507_v5/coverage_by_month.tsv`

Ключевые цифры coverage v5:

- Исходная аудио-папка: `2026-03-09--26`
- Всего аудио: `64 867`
- Исключены из ASR: `35` звонков менеджер-менеджер
- Actionable-звонки: `64 832`
- ASR done: `64 832 / 64 832`
- Full R+A: `64 832 / 64 832`
- Missing ASR actionable: `0`
- Missing full R+A actionable: `0`

Важно: полное техническое покрытие не означает, что все старые анализы уже идеальны. Сейчас отдельно идет слой улучшения качества: no-live/voicemail/IVR/virtual secretary/ASR garbage, Claude audit, consensus, staged backfill и reanalyze.

## Главный принцип уборки

Не удалять старые batch-папки и промежуточные БД до появления канонического master-layer.

Сначала нужно собрать единую базу истины, проверить ее против coverage v5, сохранить manifest и только затем архивировать или удалять старые прогоны.

Причина: сейчас результат распределен по многим batch-БД и папкам. Если удалить их до master-сборки, можно потерять трассировку: откуда взялась конкретная расшифровка, какой ASR-вариант был выбран, какой Resolve/Analyze является финальным и какие quality-fixes уже применены.

## Цель

Создать единый чистый слой проекта:

- одна каноническая БД по всем звонкам;
- один manifest по аудио и статусам обработки;
- понятная структура актуальных runtime/export/quality артефактов;
- архив старых прогонов;
- dry-run список потенциального удаления;
- нулевой риск потери данных.

## Этап 1. Инвентаризация перед уборкой

1. Зафиксировать текущие размеры основных папок.
2. Зафиксировать список всех SQLite/DB файлов.
3. Зафиксировать все runtime-папки с датами, назначением и размером.
4. Отметить активные папки, которые нельзя трогать.
5. Отметить архивные/повторные/устаревшие папки-кандидаты.
6. Проверить, что нет активных процессов ASR/R+A/quality-backfill, пишущих в БД.

Текущая оценка размера на момент фиксации плана:

- Проект всего: около `49G`
- `2026-03-09--26`: около `24G`
- `stable_runtime`: около `15G`
- `.git`: около `2.9G`
- `telegram_exports (2)`: около `1.2G`
- `_local_archive_20260424`: около `1.1G`
- `2026-03-05-21-06-49-ч1`: около `985M`
- `2026-03-05-21-06-49-ч2`: около `984M`
- `.venv-asrbench`: около `931M`

## Этап 2. Сборка canonical master DB

Собрать новую БД, например:

`stable_runtime/canonical_master_2026_05_09/canonical_calls_master_2025_01_2026_05.db`

Требования к master DB:

- одна строка на один уникальный `source_filename`;
- все `64 867` source audio присутствуют в корпусе;
- `64 832` actionable-звонка имеют ASR + R+A;
- `35` manager-manager звонков включены как `excluded_no_asr`, а не потеряны;
- у каждой строки есть provenance: из какой batch-БД/папки взяты данные;
- есть ссылка на исходный аудиофайл;
- есть статус ASR/Resolve/Analyze;
- есть transcript fields: `transcript_text`, `transcript_manager`, `transcript_client`;
- есть transcript variants: whisper/gigaAM/merge metadata, где доступны;
- есть `resolve_json`, `analysis_json`, `quality_flags`;
- есть маркеры staged quality backfill;
- есть timestamps и duration;
- есть phone/manager/source metadata.

## Этап 3. Manifest и checksums

Собрать manifest, например:

`stable_runtime/canonical_master_2026_05_09/audio_manifest_2025_01_2026_05.csv`

Manifest должен содержать:

- `source_filename`;
- абсолютный путь к аудио;
- размер файла;
- дата/время звонка из имени;
- manager/client/phone из имени, если доступны;
- checksum или быстрый hash;
- статус: `processed`, `excluded_manager_manager`, `missing`, `duplicate`, `needs_review`;
- canonical DB id;
- provenance batch DB.

## Этап 4. Проверка master-layer

Проверить master-layer против coverage v5:

- total source audio = `64 867`;
- excluded no-ASR = `35`;
- actionable = `64 832`;
- ASR done actionable = `64 832`;
- full R+A actionable = `64 832`;
- missing ASR actionable = `0`;
- missing R+A actionable = `0`;
- дубликаты объяснены;
- нет строк без provenance;
- нет строк без source audio path, кроме явно исключенных/архивных случаев.

## Этап 5. Safe archive plan

После успешной проверки master-layer подготовить dry-run архивирования.

Кандидаты на архивирование:

- старые ASR-only batch folders;
- старые `before_*` backup DB внутри batch-папок;
- повторные monthly batch-папки;
- старые `sales_master_export_*`, если есть финальные актуальные версии;
- `external_m1_*` после подтверждения импорта;
- `ab_tests`, `benchmarks`, старые pilot-прогоны, если они не нужны для разработки;
- почти пустые `messages(34)` и `messages(35)`, если там только `index.html` и аудио уже перенесено;
- старые coverage/gap отчеты, если они зафиксированы в документации и не нужны как рабочие артефакты.

Не удалять сразу. Сначала переместить в архивную зону или подготовить список удаления.

Возможная структура архива:

`_local_archive_after_canonical_master_20260509/`

В архиве сохранить:

- README с причиной архивации;
- manifest перемещенных папок;
- размеры;
- дату;
- ссылку на canonical master DB, которая заменила старые прогоны.

## Этап 6. Delete dry-run

Перед физическим удалением создать файл:

`stable_runtime/canonical_master_2026_05_09/delete_candidates_dry_run.tsv`

Колонки:

- path;
- size;
- type;
- reason;
- replacement_artifact;
- safe_to_delete: yes/no;
- requires_manual_approval: yes/no.

Физическое удаление только отдельным подтвержденным шагом.

## Что нельзя трогать без отдельного backup

- `2026-03-09--26` — основная аудио-папка.
- Текущая рабочая R+A БД: `stable_runtime/ra_missing_all_20260506/ra_missing_all_20260506.db`.
- Backup этой БД после quality backfill.
- Актуальные quality-папки от `2026-05-09`.
- AMO/Tallanto runtime.
- Документы по SaaS/productization и transcript-quality планам.
- `.env` и любые токены/секреты.
- Git-историю без отдельного решения.

## Что можно рассматривать как кандидаты после master-layer

- `stable_runtime/external_m1_jan_mar_2025_asr_only_20260504`
- `external_m1_jan_mar_2025_asr_only_20260504_result_20260506_103021`
- `stable_runtime/jun_jul_aug_2025_asr_only_20260503`
- `stable_runtime/apr_may_2025_asr_only_20260502`
- `stable_runtime/sep2025_asr_only_3000_20260504`
- `stable_runtime/sep2025_asr_only_remaining_all_20260504`
- `stable_runtime/oct_nov_2025_asr_only_remaining_all_20260505`
- `stable_runtime/final_asr_tail_1526_20260506`
- старые `overnight_*` batch folders
- старые `messages28_*`, `messages29_*`, `messages30_*`, `messages31_*`, `messages32_33_*`, `messages34_*`, `messages35_*` после проверки, что их данные есть в master DB
- `stable_runtime/ab_tests`
- `stable_runtime/benchmarks`
- `stable_runtime/venv_stable.broken_20260407`
- пустые/почти пустые `messages(34)` и `messages(35)`

Это список кандидатов, не команда на удаление.

## Следующий шаг, когда вернемся к этой задаче

1. Написать скрипт `scripts/build_canonical_calls_master.py`.
2. Сделать его в режиме dry-run без записи.
3. Сгенерировать preview: сколько уникальных звонков, из каких БД, какие конфликты.
4. После чистого dry-run создать canonical master DB.
5. Сравнить с coverage v5.
6. Сгенерировать archive/delete candidates dry-run.
7. Только после ручного подтверждения архивировать или удалять старые прогоны.

## Итоговая позиция

Наводить полный порядок уже пора, потому что техническая обработка корпуса за январь 2025 - май 2026 закрыта. Но безопасный порядок такой: сначала canonical master DB + manifest + validation, потом архивирование, потом удаление. Не наоборот.
