# Decisions Log

Дата обновления: 2026-05-23

Назначение: фиксировать принятые решения, чтобы они не жили только в чатах.

## 2026-05-15

### D-001. Main пока не трогать

Решение: рабочей веткой остается текущая ветка разработки. `main` не трогать до первого реального внутреннего запуска.

Причина: проект быстро меняется, нужно сначала довести внутренний контур до использования.

### D-002. Основной путь развития

Решение: приоритет такой:

1. Единая история клиента: звонки, письма, CRM, Tallanto, мессенджеры.
2. Рабочее место менеджера: открыть клиента, увидеть историю, получить подсказку ответа.
3. Потом расширение в сторону продукта для других организаций.

### D-003. Не строить SaaS раньше внутреннего внедрения

Решение: сначала внутренний ИИ-сотрудник для компании, потом SaaS.

Критерий: 1-3 менеджера должны реально пользоваться системой ежедневно.

### D-004. Customer timeline интегрировать

Решение: `customer_timeline` не считать мусором. Его нужно интегрировать как целевой read-only слой истории клиента.

Ограничение: сначала coverage-аудит и preview, потом основное чтение истории. Live-запись в AMO не должна зависеть от непроверенной полноты timeline.

### D-005. Бюджет хранить диапазонами

Решение: бюджет клиента хранить не только текстом, а диапазоном плюс optional комментарий.

Причина: диапазон можно фильтровать в AMO, комментарий сохраняет сложные случаи.

### D-006. Тестовый AMO не создаем

Решение: отдельный sandbox AMO сейчас не создается.

Компенсация:

- fake-тесты;
- dry-run;
- snapshot;
- rollback;
- readback;
- первый реальный микропилот на 1-5 сделках.

### D-007. AMO rollback обязателен до расширения live-write

Решение: до новых live-записей в AMO нужен pre-write snapshot и rollback.

Минимум:

- сохранить старые значения;
- откатывать только если текущее значение равно нашему записанному значению;
- не стирать ручные правки менеджера;
- поддерживать 429/5xx retry и resume.

### D-008. Структурные возражения пока не писать в AMO JSON-полем

Решение: структурные возражения хранить во внутренних preview/audit/export артефактах.

В AMO пока остается человекочитаемое поле `AI-актуальные возражения`.

### D-009. Реализация текущего ТЗ строго последовательно

Решение: порядок:

`G -> A -> PBF -> B -> C -> D -> E`

Параллельные крупные реализации не запускать. Субагентов использовать только внутри одного блока.

### D-010. Правда проекта должна быть в файлах, не в чатах

Решение: актуальные решения и состояние фиксировать в:

- `docs/CURRENT_STATE.md`
- `docs/DECISIONS_LOG.md`
- `docs/ROADMAP.md`
- `docs/RUNBOOK.md`
- audit packs

### D-011. Глобальные Codex skills установлены

Решение: установлены официальные skills:

- `security-best-practices`
- `security-threat-model`
- `security-ownership-map`
- `pdf`
- `jupyter-notebook`
- `cli-creator`

Python-зависимости skills установлены в отдельную среду:

`/Users/dmitrijfabarisov/.codex/skill-venv/bin/python`

### D-012. Основной рабочий процесс

Решение: использовать цикл:

`аудит -> ТЗ -> реализация -> тесты -> audit pack -> коммит`

Исключения допустимы только для мелких безопасных правок и read-only проверок.

### D-013. Блок G/A выполнен без live-запуска

Решение: git-границы зафиксированы отдельной картой, а AMO live-write теперь должен сохранять pre-write snapshot до PATCH.

Откат вынесен в отдельный скрипт и требует отдельный token:

`ROLLBACK_DEAL_AWARE_AMO_FIELDS`

Live-запись и реальный rollback в рамках реализации не запускались. Первый реальный микропилот остается ограниченным 1-5 сделками и требует отдельного подтверждения.

### D-014. Одиночный post-backfill звонок не должен дублироваться в хронологии

Решение: если у контакта только один содержательный звонок, отдельное поле `Хронология общения (последние 5 касаний)` остается пустым.

Причина: краткая история уже содержит компактный вывод, а хронология с одним звонком дублировала длинный пересказ последнего разговора.

Если содержательных звонков два и больше, хронология сохраняется.

### D-015. Коммерческие AMO-поля добавляются как optional

Решение: новые коммерческие поля deal-aware считаются необязательными до создания и проверки AMO-полей.

Поля:

- `AI-бюджет диапазон`;
- `AI-бюджет комментарий`;
- `AI-чувствительность к цене`;
- `AI-интерес к скидке`.

Старые 12 deal-aware полей остаются обязательными. Строгий режим `--require-commercial-fields` может специально блокировать dry-run/live, если новых полей нет в AMO-каталоге.

### D-016. Структурные возражения пока только внутренние

Решение: структурный список возражений сохраняется в preview/audit/export артефактах, но не пишется в AMO payload.

Причина: менеджеру в AMO нужен человекочитаемый текст, а JSON-поле станет полезным только когда появится потребитель: бот, аналитика РОПа или UI менеджера.

Поле `AI-актуальные возражения` остается строкой и сохраняет старую совместимость.

### D-017. Каталог вопросов подключается к deal-aware gate только опционально

Решение: deal-aware quality gate умеет принимать `question_catalog_source_index`, но без него старое поведение не меняется.

Индекс связывает `call_id` с темами, сервисными категориями, статусами политики и режимом ответа бота.

Первые блокировки ограничены самыми рискованными конфликтами: сервисная тема плюс продажный следующий шаг, manager-only тема плюс автономное действие, платежная тема плюс повторный платежный шаг, чувствительные темы плюс обещание решения.

### D-018. Customer timeline становится целевым read-only слоем истории

Решение: `customer_timeline` не удаляется и не считается мусором. Он становится целевым read-only источником истории клиента, но включается по стадиям.

В блоке E добавлен общий context provider и coverage-аудит. Stage 4 preview может показывать timeline-контекст только по явному флагу.

Timeline-контекст не добавляется в AMO payload, не входит в `DEAL_AI_FIELDS` и не является обязательным для Stage 6/live writeback.

Перед переводом в основной источник чтения нужен coverage-аудит по реальным deal-aware телефонам.

### D-019. Customer timeline локально наполнен, но флаги пока не включаем

Решение: после локального наполнения customer timeline по 100 группам телефонов/сделок флаги `timeline_preview_enabled` и `timeline_primary_read_enabled` не включаются.

Факты:

- локальная DB создана вне `stable_runtime`;
- покрытие выборки стало 100/100;
- `ready_for_preview`: 18/100;
- `needs_manual_review`: 82/100.

Причина: история технически собрана, но большая часть выбранной сложной выборки содержит deal-aware риски, AMO/Tallanto-расхождения, платежные и ручные проверки. Нельзя выдавать это менеджеру как полностью готовую историю без разбора причин.

Следующий шаг: разобрать 82 причины ручной проверки, отделить реальные проблемы данных от слишком строгих правил аудита, затем расширять локальный импорт на все 709 сделок и отдельную чистую контрольную выборку.

### D-020. Mango update 2026-05-16 переключает runtime только после зелёных gate

Решение: новые Mango-звонки не добавляются в старую canonical DB напрямую. Вместо этого создаётся новая версионированная база:

`stable_runtime/canonical_master_20260516_after_mango_update_v1/canonical_calls_master.db`

Факты:

- к принятому слою 2026-05-10 добавлено 268 terminal-звонков;
- 3 строки остались в ручном pending;
- итоговый actionable-корпус: 65 100 звонков;
- missing ASR: 0;
- missing Resolve+Analyze: 0;
- phone-chain слой пересобран в `stable_runtime/insight_readiness_report_after_mango_update_20260516_v1`;
- active export переключён на `stable_runtime/sales_master_export_20260516_after_mango_update_v1`;
- CRM quality gate зелёный;
- AMO writeback queue показывает 0 безопасных строк, ожидающих live-записи.

Отдельно зафиксирован класс ошибки: stale next-step должен блокироваться до попадания строки в AMO-ready export. Для этого export-слой передаёт detector-у alias `Дата последнего свежего звонка`, а не только внутреннее поле `Последний свежий звонок`.

Live-запись в AMO/Tallanto в рамках этого обновления не выполнялась.

### D-021. Для клиентских артефактов обязателен semantic_pass

Решение: для базы знаний, Telegram/email-черновиков, CRM/AMO/Tallanto-текстов и любых клиентских ответов `quality_passed=true` больше не считается финальной готовностью.

Новые статусы:

- `formal_pass` - структура, сборка и тесты прошли;
- `semantic_pass` - смысловая проверка прошла;
- `pilot_ready` - можно показывать сотруднику в рабочем пилоте;
- `production_ready` - можно использовать шире внутреннего пилота.

Правила смысловой проверки зафиксированы в:

`docs/SEMANTIC_REVIEW_RULES.md`

### D-022. Telegram-боту нужен слой DialogueMemory, а не только rewriter

Решение: следующий этап качества Telegram-ботов строится вокруг структурной памяти текущего диалога и классов ошибок.

Новые артефакты:

- `docs/TZ_DIALOGUE_MEMORY_AND_FAILURE_SKILLS_2026-05-23.md`;
- `docs/BOT_FAILURE_CLASSES_REGISTRY.md`;
- skill `/Users/dmitrijfabarisov/.codex/skills/bot-failure-class-review/SKILL.md`;
- модуль `src/mango_mvp/channels/dialogue_memory.py`.

Причина: targeted/smoke-прогоны показали, что точечные guards и rewriter улучшают безопасность, но не дают устойчивой памяти на 2-3 ходе. Бот должен явно хранить известные слоты, последний прямой вопрос, обещания, стадию продажи и P0-флаги.

Ограничения:

- кросс-сессионная долговременная память клиента пока не включается;
- active_brand задаётся каналом и не меняется памятью;
- P0/brand/fact guards остаются hard gate;
- `v8_targeted16` считается dev-сигналом, не финальным честным holdout.

## 2026-05-23

### D-022. Current runtime переключён на Mango update 2026-05-21 v4

Решение: текущая точка правды для звонков и AMO-ready слоя:

- canonical DB: `stable_runtime/canonical_master_20260521_after_mango_update_v1/canonical_calls_master.db`;
- active export: `stable_runtime/sales_master_export_20260521_after_mango_update_v4_runtime_acceptance`;
- pointer: `stable_runtime/CANONICAL_EXPORT.txt`;
- machine-readable contract: `stable_runtime/CURRENT_RUNTIME.json`.

Факты текущего runtime:

- actionable звонков: `65 939`;
- missing ASR: `0`;
- missing Resolve+Analyze: `0`;
- AMO-ready после CRM quality gate: `2`;
- safe writeback pending: `0`.

Старые runtime/export/quality/audit версии можно удалять только через manifest и перенос в корзину. Почтовый архив `_external_handoffs/mail_archive_2026-05-12` не удаляется: он потенциально нужен для будущей единой истории клиента.

### D-023. Cleanup batch 2: старые промежуточные артефакты вынесены в корзину

Решение: старые промежуточные export/deal-aware/KB/customer-timeline/audit-inbox артефакты, не участвующие в текущем runtime, вынесены в корзину.

Manifest:

`docs/RUNTIME_CLEANUP_BATCH2_MANIFEST_2026-05-23.md`

Корзина:

`~/.Trash/mango_cleanup_batch2_intermediate_artifacts_20260523_011201`

После переноса текущий runtime проверен: `validation_ok=true`, `blocked=0`, missing ASR/R+A = `0/0`.

### D-024. Единая рабочая папка аудиозаписей

Решение: создан единый audio working store:

`product_data/audio_working_store_20260523_v1/`

Контракт:

`docs/AUDIO_WORKING_STORE_CONTRACT_2026-05-23.md`

Факты:

- source rows из текущей canonical DB: `65 974`;
- привязано аудио: `65 974`;
- уникальных аудио по SHA-256: `65 974`;
- точных дублей по SHA-256: `0`;
- отсутствующих source files: `0`;
- size mismatch: `0`;
- неожиданных расширений: `0`;
- материализация: hardlink, без копирования байтов и без удаления исходников;
- `by_filename/` содержит ссылки с исходными именами файлов для совместимости со старыми сценариями.

Старые аудио-папки пока не удаляются. Следующий безопасный этап - переключить новые скрипты на `product_data/CURRENT_AUDIO_WORKING_STORE.txt` / manifest и только после этого отдельным manifest удалить старые исходные папки.

Для базы знаний добавлен исполняемый gate:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/run_kb_semantic_review.py \
  --release-dir product_data/knowledge_base/kb_release_20260520_v6_3_team_answers \
  --out-dir audits/_inbox/<block>/semantic_review
```

Причина: прошлые итерации проходили формальные тесты, но могли содержать смысловые ошибки: неденежные числа как цены, разорванные диапазоны, брендовые утечки, старые справочные формулировки или машинный текст в клиентском ответе.

Следствие: Codex не должен писать "готово к использованию" для клиентского слоя без `semantic_pass=true`.

### D-026. Автономность Telegram-бота включается только через матрицу и проверенные факты

Решение: целевое направление - два отдельных Telegram-бота/аккаунта для Фотона и УНПК МФТИ. Бот должен постепенно становиться автономным ИИ-сотрудником продаж, но автономность включается только по явным правилам.

Правила:

- по умолчанию бот осторожен: если тема не входит в матрицу автономности или есть сомнение, он делает черновик для менеджера или передает вопрос менеджеру;
- проверенный факт - это не оценка модели, а явный флаг в базе знаний/контексте: факт должен быть `client-safe` и актуальным;
- если тема зелёная, факт проверен и P0-рисков нет, безопасный черновик можно повышать до `bot_answer_self_for_pilot`;
- если точного факта нет, бот должен дать полезный общий ответ без точных обещаний и задать безопасные уточняющие вопросы, а не только писать “менеджер уточнит”;
- если в одном сообщении есть несколько тем и хотя бы одна P0/high-risk тема, весь вопрос уходит менеджеру; безопасную часть можно подготовить только как черновик;
- Фотон и УНПК МФТИ не смешиваются в одном клиентском ответе;
- вопросы вне темы обучения в v1 мягко возвращаются к теме активного бренда, без ответов как универсальная нейросеть.

Артефакт: `docs/TELEGRAM_BOT_AUTONOMY_MATRIX_V1_2026-05-21.md`.

Причина: цель проекта - разгрузить отдел продаж и вести клиента к покупке, но без риска выдуманных фактов, брендовых смешений, юридических обещаний или автономных ответов на незнакомые темы.

### D-027. Telegram-пилот работает двумя публичными ботами и только в read-only CRM/Tallanto

Решение: текущий пилот - два отдельных публичных Telegram-бота:

- Фотон: `@foton_intellegence_bot`;
- УНПК МФТИ: `@mipt_AI_bot`.

Факты:

- каждый бот отвечает только за свой бренд;
- база знаний: `product_data/knowledge_base/kb_release_20260520_v6_3_team_answers`;
- bot-pack: `product_data/knowledge_base/kb_release_20260520_v6_3_team_answers_bot_pack`;
- AMO/Tallanto/CRM используются только read-only через серверный контур;
- live-write в AMO/Tallanto/CRM запрещён без отдельного подтверждения.

### D-028. Gold-ответы v3 - слой качества, а не дословный скрипт

Решение: `GOLD_ANSWERS_v3` используется как эталон структуры, тона и границ ответа.

Ограничения:

- не копировать gold-ответы механически;
- не считать gold-ответы самостоятельным источником фактов;
- факты для клиентского ответа должны идти из актуальной базы знаний/контекста с client-safe флагами.

### D-029. Следующий инженерный блок - единый журнал пилота и диалоговая стратегия

Решение: текущий главный инженерный блок:

`docs/TZ_TELEGRAM_PILOT_JOURNAL_AND_DIALOGUE_STRATEGY_2026-05-23.md`

Цель:

- каждый ответ бота должен быть объяснимым через единый journal;
- ежедневный отчёт должен давать владельцу/РОПу не только счётчики, но и очереди смысловой проверки;
- сотрудники должны размечать ответы структурно;
- бот должен отвечать по-человечески: прямой ответ, память контекста, один следующий шаг.

### D-030. v8-тесты использовать только после preflight

Решение: v8 dynamic simulator является важным тестовым контуром, но не источником правды.

Порядок:

1. preflight актуальности фактов;
2. `v8_targeted16`;
3. статичные `v6/v5`;
4. полный v8 отдельным длинным прогоном с `--resume` и сохранением полных транскриптов.

Любой FAIL разбирается по категориям: ошибка бота, ошибка базы знаний, ошибка judge/test, новый regression gate или ручной контроль.

### D-025. Старые аудио-копии перенесены в корзину после SHA-проверки

Решение: после переключения runtime на `product_data/audio_working_store_20260523_v1/` старые mp3-копии аудиозаписей вынесены в корзину macOS, а не удалены безвозвратно.

Manifest:

`docs/AUDIO_WORKING_STORE_CLEANUP_MANIFEST_2026-05-23.md`

Машинные отчеты:

- `docs/AUDIO_WORKING_STORE_OLD_AUDIO_MOVED_2026-05-23.csv`
- `docs/AUDIO_WORKING_STORE_OLD_AUDIO_CLEANUP_SUMMARY_2026-05-23.json`

Корзина:

`/Users/dmitrijfabarisov/.Trash/MangoAnalyse_audio_cleanup_20260522T230404Z`

Факты проверки:

- проверено старых audio-кандидатов: `137025`;
- непокрытых SHA-256: `0`;
- current canonical DB: `65974` ссылок на новый audio store, `0` ссылок на старые audio-папки;
- `CURRENT_RUNTIME`: `validation_ok=true`, `blocked=0`, missing ASR/R+A = `0/0`.

Следствие: рабочим источником аудиозаписей является `product_data/audio_working_store_20260523_v1/`. Старые audio-пути не должны использоваться в новых скриптах; если нужен откат, файлы можно вернуть из корзины по manifest. Дополнительно очищены две малые batch audio-папки 16 мая: `mango_incremental_4_asr_ra_20260516_v1/audio` и `mango_new_21_asr_ra_20260516_v1/audio`.

### D-031. Batch 3 cleanup: старые runtime/export/ROP-AMO артефакты перенесены в корзину

Решение: после проверки текущего `CURRENT_RUNTIME` и сравнения старого/нового canonical слоя устаревшие runtime/export/ROP-AMO артефакты перенесены в macOS Trash, не удалены безвозвратно.

Manifest:

- `docs/RUNTIME_CLEANUP_BATCH3_MANIFEST_2026-05-23.md`
- `docs/RUNTIME_CLEANUP_BATCH3_MOVED_2026-05-23.csv`
- `docs/RUNTIME_CLEANUP_BATCH3_SUMMARY_2026-05-23.json`

Корзина:

`/Users/dmitrijfabarisov/.Trash/MangoAnalyse_runtime_cleanup_batch3_20260522T233839Z`

Факты:

- перенесено: `30` путей;
- после восстановления provenance-слоя чистая экономия: около `2.32 GiB`;
- root `АКТУАЛЬНО_*.xlsx` AMO/ROP/export файлы убраны как stale;
- live-write/readback evidence папки Stage51/100/200 сохранены, кроме superseded dry-run/failed-check пакетов без доказательной роли;
- `stable_runtime/history_remaining_excl_done_20260407` восстановлен, потому что текущая canonical DB хранит provenance-ссылки на него;
- `master_calls_ru.csv` активного export обновлён из текущей canonical DB: все `65 939` путей к аудио теперь указывают на `product_data/audio_working_store_20260523_v1`, отсутствующих файлов `0`.

Проверки после уборки:

- `CURRENT_RUNTIME.validation_ok=true`, `blocked=0`, missing ASR/R+A = `0/0`;
- `write_amo_ready_contacts.py` больше не зависит от старого root `АКТУАЛЬНО_AMO_ready.xlsx`, а по умолчанию читает активный AMO-ready CSV из `CANONICAL_EXPORT.txt`;
- targeted runtime/AMO/audio tests прошли.

### D-032. Batch 4 cleanup: `_local_archive_20260424` и старые deal-aware selector-fix слои разобраны

Решение: `_local_archive_20260424` нельзя удалять целиком, потому что `source_archives/messages(1).zip` содержит уникальные raw-аудиофайлы, которых нет в текущем audio working store. Удалены только производные старые подпапки архива. Старые тяжёлые deal-aware selector-fix Stage2-6 слои вынесены в корзину после замены единственной тестовой зависимости на маленький frozen fixture.

Manifest:

- `docs/LOCAL_ARCHIVE_MESSAGES1_ZIP_AUDIT_2026-05-23.json`
- `docs/LOCAL_ARCHIVE_MESSAGES1_ZIP_AUDIT_2026-05-23.csv`
- `docs/ARCHIVE_DEALAWARE_CLEANUP_BATCH4_MANIFEST_2026-05-23.md`
- `docs/ARCHIVE_DEALAWARE_CLEANUP_BATCH4_MOVED_2026-05-23.csv`
- `docs/ARCHIVE_DEALAWARE_CLEANUP_BATCH4_SUMMARY_2026-05-23.json`

Корзина:

`/Users/dmitrijfabarisov/.Trash/MangoAnalyse_archive_dealaware_cleanup_batch4_20260523T001335Z`

Факты:

- `messages(1).zip`: `231` mp3 + `1` html, `0/231` mp3 покрыты текущим audio store по SHA-256, поэтому zip сохранён;
- `_local_archive_20260424` теперь содержит только `source_archives/messages(1).zip`;
- перенесено в корзину: `10` путей, `1.215 GiB`;
- удалены старые deal-aware intermediate папки Phase1/Phase2 Stage2-6, но сохранены маленькие/доказательные слои `deal_aware_stage100_rop_final_20260514_v1`, `deal_aware_stage709_all_batches_20260514_v1`, `deal_aware_stage709_review_20260514_selector_fix_phase2`;
- `tests/test_deal_aware_confidence_recalibration.py` больше не читает удалённый runtime-слой, а использует `tests/fixtures/deal_aware_confidence_phase2_linked_scores.csv`.

Проверки после уборки:

- живых ссылок из `src/` и `tests/` на вынесенные deal-aware папки нет;
- confidence recalibration test проходит на frozen fixture;
- current runtime check остаётся зелёным: `validation_ok=true`, `blocked=0`, missing ASR/R+A = `0/0`.

### D-033. Почта + Customer Timeline: safety-границы марафона Э1-Э5

Решение: марафон Э1-Э5 по почте и Customer Timeline ведётся только через
staging-копию и `.codex_local`; prod-БД открывается только read-only
`mode=ro&immutable=1`, CRM получает только export package, а применение к prod
и AMO остаётся отдельным решением владельца. Принят отдельный журнал решений
Codex 2 с обязательным аудитом пакетов решений.

Журнал:

`/Users/dmitrijfabarisov/Claude Projects/Foton/2026-07-03_MARATHON_DECISIONS_Codex2.md`

Ключевые safety-уточнения:

- hard safety issue не может быть закрыт как «известное ограничение»;
- apply customer_timeline разрешён только внутри `.codex_local/staging/`;
- `mail_archive_stage2` может открываться боту только staging-only и за флагом
  default OFF;
- AMO/Tallanto в Э5-pre только через allowlist read-methods с доказательством
  `write_calls=0` и отсутствием токенов в логах;
- при конфликте документов для Э2-Э5 приоритет имеет марафонское ТЗ
  2026-07-03, а не старый план 2026-07-02.

Статус: принято после аудита subagent `Mendel` (`PASS_WITH_NOTES`, правки
включены в журнал). Реализация начинается только после repo-preflight.

### D-034. Wappi Telegram/Max history: порт только как manager-only staging-сырьё

Решение: импортёр истории Wappi Telegram/Max портируется из тега
`archive/wappi-history-555a964` копированием файлов, без merge старой ветки.
Источники фиксируются только как `wappi_telegram` и `wappi_max`; оба источника
запрещены для bot-visible памяти через `BOT_FORBIDDEN_SOURCE_SYSTEMS`.
Auto-resolver остаётся выключенным, а непривязанные сообщения уходят в
`pending_attribution`.

Почему так: старая ветка `codex/wappi-history` основана на устаревшей базе, но
содержит полезный импортёр. История Wappi может содержать ПДн и неоднозначные
привязки, поэтому до отдельной привязки и открытия она остаётся только
менеджерским/staging-слоем.

Проверка: staging dry-run собрал `1966` сообщений по 4 профилям, `amo_auto=0`,
`send_messenger=false`; apply создал только `1966` pending-конфликтов, без
`timeline_events` и `bot_context_chunks`; повторный apply дал `duplicate=1966`.
Прод-БД, CRM, Tallanto и live-бот не трогались.

Уборка: после PASS аудиторов, сохранённого full pytest и push порта локальная
ветка `codex/wappi-history` удалена. Удалённая локальная ветка указывала на
`555a9647f81d786d4d0e76a5136e55fca5fada29`; тег
`archive/wappi-history-555a964` указывает на тот же SHA. Remote-ветки и тег не
трогались.

### D-035. Marathon 2 Block 3: семейная карта только как cautious manager-only слой

Решение: семейная карта строится на staging детерминированно, без LLM и без
записи в prod/CRM. `confidence=high` выдаётся только при clean identity и одном
валидном ребёнке; shared-phone, ambiguous identity, несколько кандидатов,
инициалы, parent-like и служебные имена уходят в `needs_review`/`ambiguous` или
`excluded`.

Почему так: текущий `tallanto_student_snapshot` не содержит надёжных имён
учеников, а историческая `customer_profiles.sqlite` полезна только как слабый
источник. Ошибка “не того ребёнка” опаснее, чем отсутствие автоматической
привязки, поэтому блок fail-closed.

Проверка на staging:

- `family_links_total=8408`, `confident/high=3786`, `needs_review=3339`,
  `ambiguous=1210`, `excluded=73`;
- `event_child_attribution_v1=74236`, `opportunity_child_attribution_v1=16843`;
- `false_high_to_shared_phone=0`;
- `false_high_to_ambiguous_identity=0`;
- `false_high_to_multiple_high_children=0`;
- `false_high_to_obvious_suspicious_name=0`;
- repeat-apply идемпотентен, `quick_check=ok`, `llm_calls_total=0`.

CRM: в manager-only export package добавлен блок `Семья:`. Неуверенные связи
формулируются как “уточнить привязку”, а в сделку добавляется предупреждение о
семейной неоднозначности. Bot-visible память не открывалась.

### D-036. Mango fresh calls increment: только существующие identity и manager-only chunks

Решение: свежие Mango-звонки для марафона 2 портируются из тега
`archive/mango-call-increment-35fc5dd` копированием сборщика, без ASR/Analyze и
без merge старой ветки. Payload помечается как `existing_timeline_increment`;
нормализатор в этом режиме не создаёт новых клиентов и identity links.
`strong_unique` кладётся на существующий `customer_id`, `ambiguous/unmatched`
остаётся без `customer_id` и получает `pending_attribution`.

Почему так: звонок сам по себе не является авторитетной привязкой клиента.
Старый инкремент уже был построен как безопасный append к существующему
timeline, и в марафоне нельзя расширять identity-граф по одному телефону без
отдельной семейной/брендовой проверки.

Проверка на staging:

- исходный staging уже содержал `72998` `mango_processed_summary/mango_call`
  событий до `2026-06-25T13:35:43+00:00`;
- read-only сборщик взял `1031` готовый RA-звонок из локальных пакетов
  `2026-06-25T13:46:45+00:00` – `2026-07-01T15:03:25+00:00`;
- identity: `strong_unique=345`, `ambiguous=52`, `unmatched=634`;
- apply в staging создал `1031` events, `237` manager-only chunks,
  `738` conflicts, `0` customers и `0` identity links;
- bot-visible по звонкам остался закрыт: `allowed_for_bot=0`,
  `requires_manager_review=1`;
- `mango_processed_summary` добавлен в общий source-policy denylist для
  защиты от случайного `allowed_for_bot=True` в будущих импортёрах;
- повторный apply дал `duplicate=2006`, бизнес-counts не выросли,
  `quick_check=ok`.

Non-conversation звонки не получают summary/chunk/signal. Прод-БД, CRM,
Tallanto, live-бот, ASR и Analyze не трогались.

### D-037. AMO incremental: API read-only, staging apply from captured source files

Решение: AMO-инкремент в марафоне 2 читается через read-only MCP/AMO API,
но запись в основной staging выполняется из сохранённых JSONL-источников,
полученных на проверочной staging-копии. Для нестабильного MCP-транспорта
добавлен явный `--mcp-transport curl`, а для записи в уже существующую
staging-БД добавлены `--timeline-db` и явный `--allowed-root`.

Почему так: `urllib` на текущем AMO-коннекторе зависал на SSL-read без
прогресса, а повторный full fetch перед apply словил DNS timeout. Повторять
сетевую выкачку ради записи в staging рискованнее, чем применить уже
зафиксированные и проверенные JSONL, тем более они лежат под `.codex_local` и
дают детерминированный повтор.

Проверка: read-only fetch на копии прошёл двумя батчами. Первый batch взял
`5605` leads, `3861` contacts и `6000` events, но events упёрлись в `300`
страниц; второй batch от cursor взял ещё `3` leads, `1` contact и `2` events,
уже без page-cap. Apply тех же JSONL в основной staging дал `4229 + 4`
изменённых customer по карточкам и `1046` по events; repeat каждого batch дал
`changed_customer_count=0`. Итог staging: `amocrm_event=2962`,
`amocrm_snapshot/amo_contact_snapshot=14647`,
`amocrm_snapshot/amo_deal_stage=7208`, все AMO chunks
`allowed_for_bot=0`, `requires_manager_review=1`, `quick_check=ok`,
`foreign_key_check` пустой, `duplicate_source_id_groups=0`.

CRM/AMO write не выполнялся: endpoints report содержит только GET
`/api/v4/leads`, `/api/v4/contacts`, `/api/v4/events`; notes endpoint помечен
`not_used_whitelist_not_extended`.

### D-038. Nightly service v1 is a staging orchestrator, not a shell scheduler for every source

Решение: блок 5 добавляет `customer_timeline` nightly service как
staging-only обвязку над безопасными локальными incremental sources:
`mango_processed_summary` и сохранённые AMO JSONL из блока 4.3. Сервис
публикует snapshot manifest (`sqlite/-wal/-shm` sha256, `quick_check`, counts,
source counts, cursors), держит service-level lock и использует существующий
incremental lock. Launchd-пакет подготовлен как template + dry-run
install/uninstall; фактическая установка не выполняется.

Почему так: в марафоне нельзя запускать произвольные shell-команды, live API
или внешние записи из ночной службы. Tallanto, Wappi и mail уже имеют отдельные
стадийные раннеры с собственными safety-гейтами, поэтому в service v1 они
зафиксированы как disabled steps с явной причиной, а не вызываются через
универсальный shell wrapper.

Проверка: после аудиторских фиксов два финальных последовательных запуска на staging дали
`changed_customer_count=0/0` (`20260703T092146Z`, `20260703T092201Z`),
`quick_check=ok`, stable counts:
`customer_identities=20579`, `timeline_events=171733`,
`bot_context_chunks=131174`, `derived_signals=1118`; `mail_archive_stage2`,
`wappi_telegram`, `wappi_max` unsafe `allowed_for_bot=1` count = `0`.
`plutil -lint` для plist OK, install/uninstall без `--apply` только dry-run.
Prod DB, CRM, AMO write, Tallanto, live bot не трогались. Аудиторские notes
закрыты: service-lock держится до публикации manifest/latest, а service paths
и source paths валидируются относительно `allowed_root`.

### D-039. Expanded customer memory is shipped as shadow-only metadata

Решение: блок 6 переносит только безопасные части Stage01-памяти:
builder/scrub/render для `CustomerMemoryForPrompt` и shadow-runner на staging.
Память не добавляется в текст черновика и не открывается боту: новый флаг
`TELEGRAM_TIMELINE_MEMORY_EXPANDED_SHADOW` default OFF, а при включении пишет
только `metadata.customer_memory_for_prompt_shadow`.

Почему так: живой direct-path уже имеет отдельный bot-safe CRM контекст, а
расширенная память ещё требует смысловой приёмки. Поэтому в этом блоке мы
проверяем форму, источники и санитарные фильтры без влияния на ответы клиентам.
`post_layers.py` и `policy_routing.py` намеренно не трогались; старые guard-тесты
сохранены.

Проверка: shadow-run на staging по клиентам с реальными LLM email-summary дал
`62` клиентов, `55` с памятью, `7` пустых fail-closed, `safety_violations=0`,
`prompt_pii_hits=0`, `prompt_service_id_hits=0`,
`manual_review_flags=0`. Runner читает только staging DB read-only и пишет
JSONL локально в `.codex_local`, не в Foton/git. Focused pytest:
`589 passed`; полный pytest: `3969 passed, 5 skipped, 1 warning`.

### D-040. CRM v2 transfer package is operator-only until semantic-ready rows exist

Решение: блок 7 готовит CRM/SWAP-пакет только как локальный operator-only
артефакт. После смыслового аудита ready-гейты ужесточены: семейные/детские
данные, сырые email/thread-хвосты, foreign-brand маркеры, закрытый next step
при зависшей сделке и устаревшая дата next step блокируют live-ready. Итоговая
сборка CRM export даёт `candidate_rows=66`, `ready_rows=0`, `blocked_rows=66`;
`batch_ready_crm_card_candidates.jsonl` пустой.

Почему так: предыдущая версия формально проходила механические проверки, но
содержала manager-facing карточки с семейными данными, сырыми email-фрагментами
и противоречивыми next step. Для CRM-write безопаснее получить ноль ready-строк,
чем пропустить сомнительную карточку в live-update.

Проверка: D7 contract прошёл (`contract_passed=true`, `ready_jsonl_empty=true`,
`prod_db_untouched=true`). Transfer package лежит в
`.codex_local/transfer_package/marathon2_block7_20260703`, prod DB открыт только
`mode=ro&immutable=1`, sha before/after совпадает. Независимый аудитор подтвердил
PASS: ready rows отсутствуют, все рискованные строки остались в manual review.
Полный pytest: `3977 passed, 5 skipped, 1 warning`.

### D-041. Mail-summary enrichment is gated before any golden or mass LLM run

Решение: перед продолжением марафона добавлен pre-golden safety-layer для
Block 1.4. CRM target теперь берётся из `batch_ready_crm_card_candidates.jsonl`;
`pilot_20` используется только как fallback. Письма без `record.full_clean_text`
не подменяются темой, preview или старым summary: они уходят в
`summary_review_needed` с причиной `missing_full_clean_text` без LLM-вызова.
Anti-hallucination gate расширен на модельные имена, класс, предмет, дедлайн,
номера документов, реквизиты, курсы/предметы, payment/refund status, свободный
текст оплаты/возврата, обычные числовые токены, неподтверждённый `next_step` и
суммы в `amount_items`. Если quality-sanitizer меняет payload, очищенная версия
перезаписывается в `email_summary_cache_v1`.

Почему так: предыдущий слой имел formal-pass, но аудитор показал, что fallback,
пустая выжимка, неподтверждённый next step и свободный текст оплаты могли пройти
мимо review. Массовый LLM-прогон без golden-разметки запрещён, но сам
предохранитель должен быть надёжным до будущей приёмки.

Проверка: focused summary+A2 ingest `41 passed`; полный pytest:
`3988 passed, 5 skipped, 1 warning`. Plan/no-LLM на staging:
`crm_customers=99`, `review_customers=18`, `crm_review_overlap=3`,
`target_customers=114`, `target_mail_events=914`,
`missing_long_requires_summary=92`, `missing_full_text_rows=6`,
`llm_calls_total=0`. Массовый summarize/apply не запускался; semantic-pass по
качеству выжимок остаётся заблокирован до golden-набора Fable на 30 писем.

### D-042. Family graph deduplicates safe child-name variants inside one customer only

Решение: Ф1 финиша марафона добавляет детерминированный дедуп вариантов имени
ребёнка в `family_graph_v1`. Для multi-token имени первичный `name_key` теперь
строится по полному нормализованному имени, а не по одному детскому имени:
`Иванов Даниил Сергеевич` больше не схлопывается заранее в `даниил`.
Слияние разрешено только внутри одного `customer_id`: полные имена должны
совпасть минимум по двум токенам с учётом локальных детских алиасов
(`Дан/Даня/Даниил/Данил/Дениил`, `Лиза/Елизавета`) и ограниченных
опечаток/склонений фамилии. Однословное прозвище присоединяется только если
есть ровно один многословный кандидат. Неполный мост вида `Даниил Сергеевич`
не может транзитивно склеить разные полные ФИО с конфликтующими фамилиями:
такой мост помечается `excluded/ambiguous_patronymic_bridge`.
Слабые одноразовые кандидаты в шумной `identity_risk`-семье не удаляются из
БД, а помечаются `excluded/suspicious_child_name`, если выглядят как дубль
более подтверждённого ребёнка.

Почему так: регрейд архитектора показал фантомных сиблингов из опечаток и
склонений, но ложное слияние разных детей опаснее недослияния. Поэтому правило
не сливает по одной похожей фамилии, по классу/предмету или по общей теме.
`high` по-прежнему выдаётся только при единственном валидном ребёнке без
identity-risk.

Проверка: staging-пересчёт после аудиторского bridge-fix дал
`family_links_total 8408 -> 7074`, семьи с `>=4` детьми `189 -> 50`,
подозрительные `>=4` singleton-evidence семьи `175 -> 10`,
`high_in_ge4=0`, `quick_check=ok`, JSON валиден. Первичный gold v1 показал
одно расхождение без false-high; архитектор затем исправил gold до v1.1,
подтвердив, что `Даниил` и форма `Орел/Орлов Даниил` относятся к одному
ребёнку. На gold v1.1 verifier даёт `23/23`, `strict_pass=true`, `false_high=0`.

### D-043. Transfer package reports unknown git metadata instead of failing

Решение: `_read_git()` в `build_marathon2_transfer_package.py` переводит
ошибку `git` или отсутствие команды в строку `unknown`, а не роняет сборку
пакета переноса.

Почему так: пакет переноса должен быть пригоден на машине без корректного git
контекста, но это не должно создавать ложное знание о ревизии. `unknown` явно
сохраняет неопределённость и оставляет human-gate для финальной сверки.

Проверка: добавлен тест, который подменяет `subprocess.check_output` на
`CalledProcessError` и проверяет результат `unknown`.

### D-044. Email price-recall fix keeps bare "есть ли у вас" out of objections

Решение: Ф3 финиша марафона сужает `CLIENT_PRICE_INTENT_RE`: вопрос
`есть ли у вас ...` сам по себе больше не является ценовым интентом; ценовым
он становится только при явной скидке/рассрочке/льготе/Долями/многодетности.
Одновременно добавлены безопасные формы для реального ценового вопроса
`правильно я понимаю цену ... итого ...`.

Почему так: ручной recall показал, что старое правило давало ложные
срабатывания на общие вопросы ("есть ли у вас каникулы", "есть ли договор"),
а настоящий вопрос клиента про цену и итоговую сумму мог выпадать.

Проверка: тесты покрывают оба класса. На staging после пересборки:
email price-intent в голове письма `21`, accepted `20`, rejected `1`; единственный
rejected — менеджерский price-offer в теле письма, который правильно не попал
в клиентские возражения. Общая пересборка `customer_objections_v1`: `7155`
строк, email price `20`, call coverage `0.807133`, coverage gate passed.

### D-045. Manager dossier interests/pains use only explicit customer-side evidence

Решение: Ф10-досье добавляет секции `Интересы` и `Боли`, но извлекает их
только из безопасных источников: явные поля `products_of_interest`/родственные
поля в данных клиента и клиентская часть звонка `canonical_calls.transcript_client`.
`customer_opportunities.title` не считается интересом: в staging туда попадают
темы писем, акции и служебные заголовки, что создаёт шум и может выглядеть как
ложная потребность клиента. Пересказ звонка/summary тоже не используется для
болей, чтобы не принять слова менеджера за слова клиента.

Почему так: Ф10 нужен менеджеру как понятное досье, а не как максимально
широкий keyword-сборщик. Ложная боль или ложный интерес вреднее пропуска:
менеджер начнёт давить на несуществующую проблему или продавать не тот продукт.

Проверка: добавлены тесты, что интересы/боли берутся из `products_of_interest`
и `transcript_client`, но не из manager-only summary/opportunity title.
Дополнительно покрыты `record.canonical_call_id`, `call:<id>` source_id и
жёсткий запрет вывода ПДн-Excel вне `.codex_local`. Smoke на staging для 5
клиентов: `canonical_calls_loaded=65974`, `interests_total=17`, `pains_total=7`,
CRM/Tallanto/messages writes = false, Excel и summary только в `.codex_local`.

### D-046. Family gold acceptance verifies graph without tuning code to a bad gold row

Решение: Ф11 добавляет read-only verifier для `family_gold_v1.jsonl`.
Проверка сравнивает `expected_children_count`, считает false-high и понимает
`flags` как строки `ключ:значение` с матчем по префиксу ключа. Расхождение без
flags не маскируется и не чинится эвристикой: оно возвращается архитектору как
`architect_review_required`.

Почему так: gold v1 — внешний ground-truth, но если он расходится с сырьём,
подгонять семейный граф опасно. В одном случае verifier вернул
`architect_review_required`; архитектор исправил gold до v1.1, подтвердив, что
форма «фамилия впереди» относится к тому же ребёнку.

Проверка: на текущем staging с gold v1.1 `gold_rows=23`, `exact_count_ok=23`,
`count_mismatches=0`, `strict_pass=true`, `architect_review_required=false`,
`false_high=0`, `parent_name_among_children_rows=0`, `quick_check=ok`.
Подробный JSON с именами остаётся только в `.codex_local/review/f11_family_gold/`.

### D-047. F4 CRM export uses quality-based family gate and all-candidate mail enrich

Решение: CRM export больше не блокирует карточку только за наличие нормализованного
блока `Семья`. Hard-block остаётся для сомнительной family graph-строки и сырых
email/thread-фрагментов с детскими данными вне графа. Чистые упоминания ребёнка
в CRM-сводке получают мягкую пометку в `AI-предупреждение по сделке`, но не
делают карточку автоматически неготовой.

Почему так: старый гейт был написан до появления family graph и создал
структурный тупик: 66/66 кандидатов блокировались самим обязательным блоком
`Семья`. Полное снятие гейта было бы опасным, поэтому критерий заменён с
присутствия текста на качество источника.

Также CRM export пишет `all_candidates_crm_card_candidates.*`, а
`run_marathon2_mail_summary_enrich.py` использует этот список первым, чтобы
mail-enrich покрывал все 66 кандидатов, а не только `pilot_20` при пустом
`batch_ready`. Путь к canonical calls сделан fail-soft: неверный путь даёт
warning и пустой словарь, а не падение сборки пакета.

Проверка: focused pytest `52 passed`; staging mail-enrich по 66 кандидатам
создал `llm_calls_total=14`, `summary_review_needed=124`, `quick_check=ok`,
`mail_stage2_visibility_assertion=passed`. Пересборка CRM-пакета дала
`ready_rows=3` вместо `0`, `family_or_child_data_requires_review=20` вместо
`66`, idempotence passed. Прод/CRM/Tallanto/live writes = 0.

### D-048. Manager CRM cards hide pipeline statuses and trim spoken quotes

Решение: CRM-карточка для менеджера больше не показывает служебную строку
`Требуется ручная проверка модельной выжимки` в хронологии email. Вместо неё
пишется менеджерская формулировка `Письмо «<тема>»: полный текст в базе.`
Внутренние статусы пайплайна остаются диагностикой, а не текстом для AMO.

Также цитаты в секциях `Интересы` / `Боли` режутся до короткой смысловой фразы:
убираются речевые залипания, повторы соседних слов, предисловия вроде
`хотела сама вам звонить` и хвосты соседних реплик. Пример после правки:
`Нас интересует математика очная.`

Почему так: semantic-регрейд 3 ready-карточек дал `PASS_WITH_NOTES`: блокеров
безопасности не было, но менеджеру нельзя показывать внутренние статусы
конвейера, а цитата должна читаться как предложение, а не как сырой кусок
стенограммы.

Проверка: focused pytest `38 passed`; финальная пересборка
`.codex_local/staging/finish_f4_crm_export_after_mail_enrich_v6_quotesfix/`
дала `candidate_rows=66`, `ready_rows=3`, `blocked_rows=63`,
`idempotence.passed=true`, `warnings=[]`. Spot-check 3 ready-карточек:
служебная строка, raw internal statuses, fallback, маски и речевые залипания
не найдены. Прод/CRM/Tallanto/live writes = 0.

### D-049. F5 M1 bundles are lightweight inputs, not a runner

Решение: Ф5 готовит два локальных бандла для ручного запуска на M1:
`email_summary_quality_100.jsonl` для semantic-review 100 LLM-выжимок и
`memory_shadow_*` для OFF/SHADOW сравнения памяти. Скрипт не запускает M1,
не создаёт queue-файлы и не содержит флаги фактической постановки в очередь.

Почему так: по ТЗ Codex должен подготовить входы, а M1 запускает человек. Это
сохраняет контроль над подпиской/очередью и не смешивает staging-подготовку с
боевыми или прогонными действиями.

Источник для email-quality — только текущие `email_summary_cache_v1` строки
со `source_kind='llm'` в staging: сейчас их `227`, из них выбрана
стратифицированная сотня. Исторические числа 845/911 не используются как
истина для текущего Ф5-бандла, потому что они относились к более широким или
предыдущим enrichment-счётчикам.

Проверка: `f5_m1_bundles_v2` содержит `sample_count=100`, memory micro `12`,
full `20`, overlay `18` клиентов / `18` bot-safe chunks, `pii_scan=passed`.
`memory_shadow_run_commands.sh` не содержит `--execute` и `--streams-ready`.
Focused pytest `41 passed`. Прод/CRM/Tallanto/live/M1 writes = 0.

### D-050. F6 Wappi import remains a pending-attribution decision package

Решение: Ф6 не выполняет боевой Wappi-долив. На базе блока 4.1 собран локальный
decision package `.codex_local/review/f6_wappi_prod_import_package_v1/` с
манифестом, runbook, checklist и masked-очередью чатов для ручной привязки.
Текущее состояние `1966 pending_attribution` считается правильным fail-closed
результатом, а не дефектом: пока нет явной связки `chat -> customer/deal`,
история не должна становиться событием клиента или bot-visible памятью.

Почему так: Wappi не должен угадывать личность по имени, частичному телефону
или бренду. Перед production apply нужен ручной pair-файл примерно по `145`
чатам, dry-run на свежей staging-копии, spot-check linked-чатов и отдельное
разрешение владельца.

Проверка: исходный блок 4.1 дал `records_built=1966`, `wappi_telegram=1000`,
`wappi_max=966`, `linked_by_pair=0`, `linked_by_amo_auto=0`,
`pending_attribution=1966`, повторный apply `duplicate=1966`,
`wappi_events=0`, `bot_context_chunks=0`, `allowed_sum=0`, `quick_check=ok`.
Ф6-пакет содержит только инструкции и локальный masked backlog; прод/CRM/live
write = 0.

### D-051. F7 result-image matrix separates achieved mechanics from live enablement

Решение: Ф7 оформлен как матрица `образ результата -> достигнутое`, а не как
релизный статус. В отчёте отдельно помечены состояния `staging`, `package`,
`shadow`, `human gate` и `not covered`, чтобы не спутать написанную механику
с включённым ИИ-сотрудником.

Почему так: к этому моменту большая часть механики данных уже есть, но live
бот не читает память, CRM-write не выполнялся, Wappi остаётся pending, M1 не
запускался Codex, а сайт-канал марафоном не закрыт. Завышенный статус опасен:
он может привести к преждевременному включению памяти или CRM-записи.

Проверка: F7-отчёт ссылается на `2026-06-29_OBRAZ_REZULTATA...` и текущие
Marathon-2 отчёты. В нём явно написано, что нельзя говорить: память клиента
уже включена в live, CRM можно писать автоматически, Wappi готов к apply,
M1-quality доказан или сайты включены. Прод/CRM/live writes = 0.

### D-052. F9 AMO актуальность проверяется по клиенту, а не по сигналу

Решение: для топ-50 `deal_stalling` Ф9 сверяет живой AMO через открытые сделки
клиента из `customer_opportunities`, а не через сам сигнал. Причина: сигнал
`deal_stalling` хранит `customer_id`, но не является авторитетным носителем
`lead_id`.

Сверка использует только `AmoMcpClient.amo_api_get` и `read_mcp_env`, без
`crm_call.sh` и без токенов в логах. Если read-only env недоступен, результат
становится `unavailable`, а не фальшиво зелёным. Менеджерский Excel волны-0
с ПДн пишется только в `.codex_local/review/f9_amo_actuality/`.

Проверка: read-only run дал `customers_selected=50`,
`open_opportunities_checked=49`, `customers_checked=42`,
`customers_changed=0`, `snapshot_stale=false`, `errors=[]`. Excel содержит
`500` строк на листе `Зависшие факт LTV`, `90` сезонных и `109`
`Вернулись и перезвон`. Focused pytest `41 passed`, включая guard, что
Wave-0/refresh manager views нельзя писать вне `.codex_local`. Прод/CRM/Tallanto/live
writes = 0.

### D-053. F10 manager dossier is local-only and includes the full 84-client review set

Решение: Ф10-досье строится как локальная Excel-книга для менеджерской
вычитки, а не как боевой CRM-write. В список включены `66` CRM-кандидатов
из финального Ф4-пакета и `18` полных review-клиентов из
`.codex_local/review/gold/review18_full_ids.json`, всего `84/84` клиентов.

Email-выжимки для этих клиентов прогоняются только существующим
`scripts/run_marathon2_mail_summary_enrich.py`: `679` email-событий, `6`
LLM batch calls, `47` строк оставлены как review-needed и показываются в
досье как «полный текст в базе», без внутренних статусов пайплайна. Apply
пишет только staging: `enrich_existing_events=679`, `created_events=0`,
`created_chunks=141`, `mail_stage2_unsafe_chunks=0`, `quick_check=ok`.

В менеджерской книге источники показаны человеко-читаемо (`AMO снимок`,
`сводки звонков`, `письма`, и т.п.), а технические ключи, `Codex/Claude/GPT`,
`llm_fallback`, `Требуется ручная проверка модельной выжимки` и generic
`Посмотреть историю...` не попадают в листы. Разделы `Интересы`, `Боли` и
`Возражения` остаются advisory и требуют human-gate, потому что часть текста
приходит из ASR/автоматических маркеров.

Проверка: финальная книга
`.codex_local/review/dossier/2026-07-04_VOLNA1_manager_dossier_full84.xlsx`
содержит `4380` клиентов в полном сегменте, `103` family rows, `31` money
rows, `65` signals, `60` next-step rows, `210` objections, `69` interests,
`94` pains и `2099` chronology rows. Focused pytest `41 passed`. Семантический
аудит: `PASS_WITH_NOTES`; ПДн остаются только в `.codex_local`; прод/CRM/Tallanto/live
writes = 0.

### D-054. Marathon-2 final gates separate automated pass from human gates

Решение: финальный статус марафона-2 фиксируется как staging/package готовность,
а не как разрешение на прод-применение. Автоматическая часть Ф1-Ф11 доведена до
зелёных локальных гейтов, но перенос staging в prod, CRM-write, Wappi-долив,
включение памяти в live и M1-прогоны остаются отдельными решениями владельца.

Почему так: марафон сознательно не писал в prod/CRM/live. Его результат —
пакеты, staging-БД, отчёты, Excel для менеджерской вычитки и скрипты
применения/сверки. Называть это production-ready было бы опасно: часть
смысловых проверок остаётся human-gate, M1/Wappi/CRM-write требуют отдельных
решений, а перенос staging в prod не выполнялся.

Проверка: после фикса fail-soft warning для отсутствующей canonical calls DB
и guard, запрещающего писать подробный family-gold JSON вне `.codex_local`,
полный `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest tests/ -q`
дал `4027 passed, 5 skipped` за `85.33s`. Warnings ожидаемые: системный
LibreSSL и тестовый `canonical_calls_db_missing:*`. Ф11 после исправления
gold до v1.1: `gold_rows=23`, `exact_count_ok=23`, `strict_pass=true`,
`false_high=0`, `architect_review_required=false`. Прод/CRM/Tallanto/live
writes = 0.

### D-055. Transfer package points to the current canonical CRM export snapshot

Решение: финальный transfer-пакет для передачи владельцу ссылается на
`.codex_local/staging/finish_f4_crm_export_current_transfer_v1`, а не на
устаревший `.codex_local/staging/block7_crm_export_v2_final`.

Почему так: старый block7-снимок был собран до semantic/gate/mail-fix и давал
`ready_rows=0`, тогда как принятый архитектором CRM-пакет v6/v6-compatible даёт
`ready_rows=3`. Показывать владельцу оба числа без пояснения опасно: это
выглядит как конфликт результатов. Для текущей staging-БД CRM export пересобран
заново, чтобы sha staging совпадал с transfer-манифестом.

Проверка: свежий CRM export
`finish_f4_crm_export_current_transfer_v1` дал `candidate_rows=66`,
`ready_rows=3`, `blocked_rows=63`, `idempotence.passed=true`,
`timeline_db_sha256=d04e4b...`. Transfer-пакет
`.codex_local/transfer_package/marathon2_block7_20260703/` перегенерирован с
этим CRM export и теперь в `crm_package_reference.md`/`manifest.json` показывает
`ready=3`. Прод/CRM/Tallanto/live writes = 0.

### D-056. Owner-approved bot data opening is staging-only before M1 memory measure

Решение: по «ДА» владельца от 04.07 rich-память открывается боту только в
staging-БД и только через Э4б-gate: linked/non-conflicted, not blocked/pending,
known content brand (`foton`/`unpk`), `allowed_for_bot=1` и
`requires_manager_review=0`. Live-бот и prod-БД не меняются; включение памяти в
живой prompt остаётся отдельным решением после M1-замера.

Почему так: само изменение `allowed_for_bot` не достаточно — runtime/direct-path
должны уметь читать rich email/telegram/wappi chunks, а M1 должен мерить именно
этот новый слой. Unknown-brand chunks оставлены закрытыми, потому что для
двухбрендового бота allowed-память без бренда создаёт риск смешения и ложный
отчёт «бот видит», хотя prompt-фильтр её отбросит.

Проверка на staging: первый apply был остановлен аудитом, потому что rich
chunks открывались для части partial/conflicted identities. После фикса Э4б
требует `customer_identities.identity_status='strong'`, нормализует ссылки
`timeline_conflicts.entity_refs` и retract-ит ранее открытые не-openable chunks.
Исправленный apply оставил открытыми `8 751` chunks (`7 984`
mail_archive_stage2 + `767` telegram_history), `wappi_* = 0`, `6 328`
unknown-brand chunks оставлены закрытыми, `5 634` ранее открытых chunks
отозваны обратно в `allowed_for_bot=0/requires_manager_review=1`.
`candidate_review_violations_after=0`, `opened_disallowed_identity_after=0`,
`opened_unknown_brand_after=0`, `quick_check=ok`. Повторный apply обновил `0`
строк и ничего не retract-ил. После повторного аудита retract/update операции
дополнительно ограничены `tenant_id`, чтобы будущий multi-tenant запуск не
трогал соседний tenant; регрессия покрыта тестом. M1 overlay v3 собран в
`~/Yandex.Disk.localized/OpenClaw/mango_m1_f5_20260704/`:
`memory_shadow_overlay_v3.sqlite`, `quick_check=ok`, `pii_scan=passed`,
`59` chunks (`18` bot_safe_summary, `34` email_message, `7`
channel_message). Прод/CRM/Tallanto/live writes = 0.

### D-057. M1 memory prompt must measure source-policy-opened chunks, not partial memory

Решение: v3.1-промпт для M1 нельзя отдавать без кодового фикса. ON-ветка
замера должна включать E4b source-policy flags для `mail_archive_stage2` и
`telegram_history`, а runtime-reader обязан сохранять `source_system` в
bot-safe items. Иначе direct-path видит `chunk_type=email_message`, но без
`source_system=mail_archive_stage2`, режет весь email-контекст и precheck
останавливается с `0` prompt-items при ненулевых expected-hits.

Почему так: это measurement_bug, не проблема overlay. Данные и expected-hits
есть, но контракт между `bot_safe_runtime_context` и direct-path терял поле
источника. Исправление не открывает новые данные: оно применяется только к
chunks, уже прошедшим `allowed_for_bot=1`, `requires_manager_review=0`,
source-policy, brand и PII-фильтры.

Дополнительные решения по хвостам: overlay поднимается с v3.1 до v3.2, потому
что в `customer_identities.record_json.source_ref` найдены 4 клиентских
телефона вида `master_contact:+...`; v3.2 чистит только identity-source-ref и
не переписывает тексты chunks. Дубли email на чтении режутся по
`metadata.message_sha256` до `project_bot_context()`, база не мутируется.
Контроль писем B1 требует суженной правки judge prompt: subject можно считать
контекстом/темой письма, но нельзя подтверждать оплату, возврат, договор,
запись или сумму только из subject без поддержки в теле письма или структурном
поле.

Проверка: локальный pre-LLM probe на пакете OpenClaw v3.1 после source-system
фикса дал `passed=true`, включая `memory_rich_13_foton_unknown_77b96f94`.
Точечный набор
`PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q
tests/test_bot_safe_runtime_context.py tests/test_bot_safe_direct_path_context.py
tests/test_memory_measure_apparatus.py tests/test_customer_timeline_read_api.py
tests/test_marathon2_m1_bundles.py` дал `62 passed`. Прод/CRM/Tallanto/live
writes = 0.

### D-058. Nightly D v2 consumes staging-local inputs; external handoffs are prebuilt, not service roots

Решение: ночная служба D v2 не расширяет `allowed_root` на соседний worktree
`Mango analyse` и не читает `_external_handoffs`/`product_data` напрямую.
Отдельный безопасный producer собирает локальные входы в
`.codex_local/staging/nightly_dv2_sources/`, а `nightly_service` работает
только с этими staging-local JSONL/manifest-файлами.

Почему так: ТЗ требует запись только в `.codex_local/staging/**` и запрещает
prod/stable_runtime/live-write. D0-артефакты физически лежат в соседнем
`Mango analyse`, а не в текущем worktree. Если дать service прямой доступ ко
всему `Projects`, он перестанет быть проверяемым single-root процессом.

Уточнение по источникам: готовый mail stage2 после текущего курсора дал `11`
строк, но свежие архивы `2026-06-29..2026-06-30` и
`2026-06-30..2026-07-06` дают ещё `244 + 1057` сообщений без уверенной
`customer_id`. Они импортируются в staging как `mail_archive_stage2`
`pending_attribution`, `allowed_for_bot=0`, `requires_manager_review=1`,
`needs_summary_later=true`; это не память боту и не CRM-запись.

Wappi в D v2 оставлен monitor/pending-only: используется уже созданный
`block4_wappi_metrics.json`, `timeline_events` для `wappi_telegram`/`wappi_max`
остаются `0`. Mango API в D v2 — только freshness-monitor локальных
`mango_update_after_*`; nightly не делает сеть и не запускает ASR. Tallanto —
optional monitor: staging snapshot есть, но nightly-ready свежей выгрузки нет,
поэтому курсоры фиксируются как `tallanto_snapshot=2026-05-21T08:59:36+00:00`
и `tallanto_crm_call=2026-06-04T16:54:54+00:00`.

Проверка: producer собрал `1312` mail rows, первый service-run
`20260707T000403Z` прошёл `overall_status=ok`,
`latest_published=true`, `quick_check=ok`, latest sha
`94f77712f78f9de3312d8625a5cd1175db6e33e50ca6588ee02a6d095d910ce2`.
Mail-only rerun дал `changed_customer_count=0`, boundary-overlap `1`
accepted record with `write_status_counts.duplicate=1`. Прод-БД/AMO/Tallanto/
live-бот/сеть/ASR/LLM не трогались этим блоком.

### D-059. Micro v3.3 fixes contact-data claims locally; judge/fact_audit stay unchanged

Решение: клеймы вида «телефон/почта/адрес уже есть у нас/в диалоге» режутся
детерминированным post-layer guard на стороне бота. Если в текущей реплике
клиента или в client-specific факте нет подтверждённого контакта клиента,
фраза переписывается в нейтральное «Повторно указывать не обязательно —
менеджер сверит по системе». Телефон/почта учебного центра не считаются
доказательством контакта клиента.

Почему так: micro v3.2 показал реальный дефект ON-21 — бот заявил «телефон
уже есть в диалоге», хотя телефона не было ни в диалоге, ни в client-safe
фактах. `fact_audit` j4 такой тип клейма не видит (`has_unverified_claim=false`
на ходе), поэтому исправление сделано как точечный output guard, а не правка
judge. Расширение `fact_audit` — кандидат на j5, не на этот микро-заход.

Дополнительно: generic-клеймы судьи остаются вариативными. Пара OFF-09-T5
и 16-T5 может давать разные verdict при близком тексте про SohoLMS/запись,
поэтому этот класс зафиксирован как measurement variability, без нового
жёсткого gate.

Проверка: добавлены unit и direct-path regression tests: неподтверждённый
контакт переписывается; контакт из реплики клиента сохраняется; телефон центра
в фактах не разрешает клейм «контакт клиента уже есть»; no-memory рамка
«лучше начать с…» переписывается без изменения route.

### D-060. Mail link enrich is a separate staging-only step; email remains weak and bot visibility is unchanged

Решение: для 1 312 pending `mail_archive_stage2` сообщений добавлен отдельный
staging-only шаг `mail_link_enrich` после `mail_archive_incremental`. Он читает
сырой mail-archive envelope/signature по `message_sha256`, делает `strong`
только по уникальному телефону из подписи/надёжному identity-link и оставляет
email-match как `weak_email` без привязки к клиенту. Новые mail chunks всегда
создаются только `allowed_for_bot=0`, `requires_manager_review=1`; открытие
почты боту этим решением не выполняется.

Почему так: email сам по себе даёт слишком много ложных совпадений, а телефон
из тела письма может быть чужим. Поэтому телефон извлекается только из короткой
нецитированной подписи, а body/citation phone не считается strong-сигналом.
Повторный `mail_archive_incremental` сохраняет уже принятые
`mail_link_enrich/pending_reason/customer_id` по `message_sha256`, чтобы full
nightly re-run не стирал результат enrich.

Результат на staging: dry-run по 1 312 дал `strong=25`, `weak_email=5`,
`blocked=3`, `unmatched=1279`; apply обновил 1 312 событий и создал 25
manager-only chunks. Финальный service-run `20260707T015146Z` дал
`overall_status=ok`, `mail_link_enrich.target_events=0`,
`pending_without_reason=0`, `quick_check=ok`. `allowed_for_bot` не изменился:
всего `26607`, для `mail_archive_stage2` `7984` до/после.

Хвосты B2-B4: B2 phone-like проверен по реальному `text` bot-visible chunks и
`customer_identities.source_ref` в overlay v3.2 — хитов `0`; patch overlay не
нужен. B3 storage-дубли по mail source accepted, bot-read дедуп остаётся на
чтении по `message_sha256/source_id/source_ref`, БД не мутируется. M4 cursor
literal переименован: новый monitor пишет `wappi_history_pending` и удаляет
устаревший staging-cursor `wappi_history`; latest manifest больше не содержит
`wappi_history`. Prod/CRM/Tallanto/live writes = 0.

### D-061. Shadow memory is split from in-prompt memory; contour packages stay operator-only

Решение: `TELEGRAM_TIMELINE_MEMORY_SHADOW` больше не является алиасом
реального включения bot-safe CRM context в prompt. Реальный prompt включает
память только через `TELEGRAM_BOT_SAFE_CRM_CONTEXT=1` или
`TELEGRAM_TIMELINE_MEMORY_IN_PROMPT=1`; shadow/expanded-shadow собирают trace
и локальную телеметрию без вставки памяти в клиентский prompt. Локальный
shadow-runner по умолчанию пишет только счётчики/хэши/причины, без
`prompt_text`; сырой текст включается только явным `--include-prompt-text` для
локального аудита.

Почему так: K6 требовал доказать, что тень не становится боевым prompt. В
старом helper `bot_safe_crm_context_enabled()` shadow-флаг включал тот же
builder, что и `IN_PROMPT`, поэтому стенды и draft-loop могли случайно мерить
не shadow-only. Развод сделан до любых пакетов включения памяти.

Операторские пакеты: добавлены launchd-шаблоны дневных capture-driver'ов и
runbook nightly 03:30, но установка/launchctl/SWAP/live-включение остаются
ручными действиями владельца. Driver'ы dry-run по умолчанию, с lock-файлом от
двойного запуска; Mango capture не запускает ASR без явно заданного
`MANGO_CAPTURE_COMMAND_FILE`.

Проверка: plist lint OK; driver dry-run OK; install/uninstall dry-run OK;
K4 e2e staging probe `run_20260707T022446Z`: source mail rows `1312`,
service statuses `ok/ok`, `allowed_for_bot_delta=0`, re-run zero-new по
events/chunks/links/conflicts. K6 shadow-run: `75` клиентов, `66` found,
`safety_violations=0`, `raw_prompt_text_rows=0`. SWAP transfer-package
`marathon2_noch_current` пересобран на staging sha
`84466460f447073d4237c9fe724994b118c2a8f593c08ecd605d80166d62d688`,
CRM export `ready=3/blocked=63`, prod sha до/после не менялся.

Ограничение: K7 “большой экзамен на ~100 реальных live Telegram-диалогах” не
запускался, потому что локально найден только scrubbed replay на 10 Wappi-кейсов
и старые/synthetic dynamic transcripts; свежего набора 100 live Telegram
диалогов последних 2-3 недель в `.codex_local`/рабочем дереве не найдено.
Подмена источника запрещена; нужен отдельный capture/export реальных диалогов.

### D-062. Owner accepted SWAP+memory risk, but live execution requires a matched runtime

Решение владельца зафиксировано дословно: «SWAP + включение памяти, трафик на
публичного бота ≈ 0, красный micro v3.4 и дыра черновиков известны».

Гейт на будущее: до подачи трафика на публичного бота (реклама/основной канал)
нужно закрыть развилку `draft_for_manager`: сейчас менеджер не видит такие
черновики как отдельный рабочий контур, и live public bot может отправлять
клиенту итоговый текст независимо от route.

Исполнение SWAP 2026-07-07 остановлено предполётом, несмотря на совпадение DB
sha: live bot работает из `/Users/dmitrijfabarisov/Projects/Mango_main_intent_ff`
на ветке `codex/adr003-semanticframe-migration`, а transfer-package
`marathon2_noch_current` собран из worktree
`/Users/dmitrijfabarisov/Projects/Mango_email_pipeline_restore` на другом HEAD.
Текущий live-start script не фиксирует `TELEGRAM_TIMELINE_MEMORY_IN_PROMPT=1`
и не задаёт явный путь к Customer Timeline DB. Поэтому env-включение памяти
без синхронизации live worktree/commit/runbook могло стать no-op или смешать
несовместимые код и данные. DB-файлы не подменялись, live bot не
останавливался.

### D-063. Dev schedule runs as Codex automations before launchd

Решение: до стабильности Customer Timeline pipeline дневные водители и
ночной кладовщик запускаются задачами Codex по расписанию. Launchd-переезд для
каждой задачи разрешается после 3-5 подряд чистых запусков без ручных правок:
`re-run=0`/идемпотентность, нет stop-причин, нет записей в prod/AMO/Tallanto,
нет ASR и массовых LLM.

Единый вход для Codex-задач и будущих plist-шаблонов:
`scripts/run_customer_timeline_codex_task.py`. Он запускает штатный driver,
пишет полный лог в `.codex_local/staging/codex_dev_tasks/` и 5-строчную
обезличенную сводку в `/Users/dmitrijfabarisov/Claude Projects/Foton/_daily/`.
При аномалии останавливается только текущая задача.

Состав задач:
`mail-capture` вызывает mail driver в staging apply; `mango-capture` делает
безопасный dry-run/accounting, пока не задан `MANGO_CAPTURE_COMMAND_FILE`;
`tallanto-api-capture` fail-closed до явного
`TALLANTO_API_CAPTURE_ENABLED=1`; `nightly-warehouse` вызывает staging-only
nightly service с Dv2 config. Live/prod/CRM/ASR не входят в эти задачи.

### D-064. Publish snapshot v3 becomes codex-executable, but flip is gated by live-reader preflight

Решение владельца: собрать `scripts/publish_snapshot/` как исполняемую
автоматизацию runbook-v3 для ночной публикации Customer Timeline snapshot.
Инструмент codex-executable, но `flip --execute` разрешён только после зелёного
машинного preflight: чистые reader worktree, stop/start команды из конфига,
reader-smoke реального live-кода, WAL checkpoint с `-wal=0`, backup+rollback.

Почему так: предыдущий SWAP был остановлен из-за несовпадения runtime
(`Mango_main_intent_ff`) и transfer-package (`Mango_email_pipeline_restore`).
Новый tooling должен ловить ровно этот класс риска до stop/start и до подмены
стабильного пути.

Проверка 2026-07-07: tooling и тесты добавлены; reader-smoke live-кода против
staging прошёл. Реальный preflight пакета `marathon2_noch_current` заблокировал
публикацию: live reader worktree `/Users/dmitrijfabarisov/Projects/Mango_main_intent_ff`
грязный и используется соседней задачей. DB flip не выполнялся, live bot не
останавливался, prod DB не подменялась.

### D-065. Wappi daily capture/resolver writes only manager-only staging records

Решение: Wappi Telegram/Max дневной захват по 4 профилям
(Фотон/УНПК × Telegram/Max) выполняется read-only через Wappi GET и AMO GET
resolver. Статические пары draft-loop не используются в этом проходе; strong
match = AMO exact Telegram ID или Max phone + ровно одна активная сделка
нужного бренда + существующий customer/opportunity в timeline. Неуверенные
чаты остаются `pending_attribution`.

Прогон 2026-07-07: в staging влито `1469` Wappi events/chunks
(`wappi_telegram=1329`, `wappi_max=140`), все chunks manager-only
(`allowed_for_bot=0`, `requires_manager_review=1`). Open Wappi pending остались
только с конкретными fail-closed причинами; старый `draft_loop_pair_missing=0`,
open-conflict по уже влитому event = `0`, `quick_check=ok`. Внешние записи:
client sends=0, CRM/AMO write=0, Tallanto write=0.

### D-066. First publish snapshot attempt is blocked by dirty live reader worktree

Попытка первой публикации Customer Timeline snapshot 2026-07-07 остановлена на
`scripts/publish_snapshot/preflight.py`. До `build_snapshot`, stop live bot,
flip и e2e-пробы дело не дошло.

Причина: live reader worktree
`/Users/dmitrijfabarisov/Projects/Mango_main_intent_ff` не чистый
(`?? product_data/telegram_dynamic_test_sets/adr003_kombo_factsel_veto_masker_ed59692b_20260707.jsonl`,
`?? product_data/telegram_dynamic_test_sets/adr003_kombo_factsel_veto_masker_ed59692b_20260707_README.md`,
`?? tasks/_running/2026-07-07_TZ_KOMBO_zahod_minus_regex_odna_para_dlya_D1.md`).
По runbook-v3 это hard stop: читатель, который будет останавливаться и
стартовать после flip, должен быть проверяемо чистым.

Перед стопом усилен publish tooling: в конфиг добавлены 3+2 контрольных клиента
с эталонными счётчиками, а `reader_smoke.py` теперь сравнивает totals
(`events_total`, `bot_context_chunks_total`, `allowed_chunks`,
`review_required_chunks`, `derived_signals_total`), а не только `found=true`.
Prod DB не подменялась, live bot не останавливался.

### D-067. Publish preflight allows data-only untracked files, but blocks without off-disk rollback backup

Решение владельца: для публикации snapshot live-reader worktree не обязан быть
абсолютно пустым. Блокируют только modified/staged tracked-файлы и untracked в
кодовых путях `src/`/`scripts/`. Untracked в data/service paths
(`product_data/telegram_dynamic_test_sets/`, `tasks/`, README и т.п.) не
блокируют; они фиксируются в preflight-отчёте и в `build_manifest.json` как
`live_worktree_untracked`.

Повтор preflight 2026-07-07 подтвердил: текущие три untracked файла в
`/Users/dmitrijfabarisov/Projects/Mango_main_intent_ff` являются data/service
untracked и больше не блокируют публикацию. Новый hard stop: обязательный
rollback backup на другом filesystem не настроен (`backup_root_missing`), а на
машине сейчас виден только основной пользовательский filesystem
`/System/Volumes/Data`; `/Volumes` содержит только ссылку `Macintosh HD -> /`.
Поэтому `build_snapshot`, stop live bot, `flip` и e2e-проба не запускались.

### D-068. Owner removes separate-filesystem backup requirement for snapshot flip

Письменное решение владельца от 2026-07-07: требование «backup на отдельный
filesystem» снято. Для первой публикации snapshot достаточно локального бэкапа
в `prod_backups/` на том же диске с обязательной sha256-верификацией после
копирования, плюс локальная копия этого бэкапа в
`~/Yandex.Disk.localized/OpenClaw/prod_backups/`; облачная синхронизация
Yandex/OpenClaw считается второй точкой.

Новый hard stop по публикации: запись в prod запрещена без проверенного
локального backup sha. Отправка клиенту остаётся абсолютным стопом. Tooling
обновлён: `preflight` проверяет оба backup-root и свободное место, `flip`
создаёт и sha-проверяет обе копии до удаления sidecar-файлов и атомарной
подмены prod DB.

### D-069. Customer Timeline snapshot #1 is published to stable prod path

Решение владельца от 2026-07-07 выполнено: первый snapshot Customer Timeline
опубликован по стабильному prod-пути
`product_data/customer_timeline/customer_timeline_prod_20260621/customer_timeline.sqlite`.

Опубликованный snapshot:
`prod_snapshots/prod_20260707_200020/customer_timeline.sqlite`, sha256
`836c8713ff7292f9a80dfdaf03d85bde0101fb9ebdb447fb20589a00473bf57c`.
Перед подменой создан проверенный rollback backup старого prod:
`prod_backups/pre_flip_backup_2026-07-07T170221.347644Z0000/customer_timeline.sqlite`,
sha256 `ef9ef249b4192b768cd1eb826f6df20514994539a3911f9aeee19bbc295d03c8`;
такая же sha256-проверенная копия лежит в
`~/Yandex.Disk.localized/OpenClaw/prod_backups/pre_flip_backup_2026-07-07T170221.347644Z0000/`.

Post-flip smoke: prod `quick_check=ok`; real live-reader smoke rc=0; 5/5
контрольных клиентов совпали с manifest counts; prod sha256 совпадает с
snapshot sha256. `flip.py` завершился rc=1 из-за timeout ожидания
долгоживущего `start_command`, но live bot screen/process поднят и держит новую
DB. Tooling исправлен отдельным коммитом: timeout команд теперь возвращается
структурированным JSON (`rc=124`, `timed_out=true`), а не роняет отчёт.

E2E через Telegram не выполнялся, потому что отправка клиенту остаётся
абсолютным stop. Вместо этого использованы non-send проверки: live-reader smoke,
process/screen presence, prod DB lsof/sha/quick_check. В логах public bot после
старта виден `getUpdates Conflict`; это операционный риск Telegram-поллера, но
не откатывает опубликованный Customer Timeline snapshot.

### D-070. Wappi draft-loop wrote first 5 AMO manager draft notes after snapshot #1

Решение владельца от 2026-07-07 выполнено: после публикации snapshot #1
записаны 5 Wappi→AMO заметок-черновиков в карточки AMO через AI Office notes
endpoint. Run ID: `wappi_live_notes_20260707_1717`.

Границы прогона: `DRAFT_LOOP_AUTO_RESOLVER=0`, основной
`~/.mango_secrets/draft_loop_pairs.json` не менялся; использованы временные
pairs/profiles only for five selected chats. Wappi calls были только GET с
`mark_all=false`; Wappi/client sends=0. После AMO-write повторный Wappi readback
показал, что во всех 5 чатах последняя реплика всё ещё входящая.

AMO readback подтвердил текущие note_id: `471856971`, `471856973`, `471856975`,
`471856977`, `471856979`. Все заметки содержат маркер
`ЧЕРНОВИК БОТА, не отправлено`, brand, timestamp, run-id и safety flags.

Ограничение: для выбранных пяти старых unanswered Wappi-чата
`memory_hits=0` при включённом `TELEGRAM_TIMELINE_MEMORY_IN_PROMPT=1`; в
опубликованной DB для этих lead_id не нашлось bot-visible контекста. Поэтому
этот проход доказывает безопасный AMO note write и client=0, но не доказывает
пользу памяти. Семантический статус отчёта: `PASS_WITH_NOTES`.

### D-071. Wappi memory-positive draft notes confirm runtime memory path, with retro caveat

Диагноз первой пятёрки из D-070 уточнён SQL и runtime-проверкой: у lead_id
`48679534`, `49972887`, `48668212` есть по одному raw allowed
`bot_safe_summary`, но они не попали в prompt, потому что runtime builder
отфильтровал их по brand-scope/`unknown`; у `49258173`, `49398199` нет
allowed chunks. Поэтому `memory_hits=0` был fail-closed поведением, а не
поломкой контура.

После этого записана вторая пятёрка Wappi→AMO заметок-черновиков, выбранная по
строгому runtime-критерию `build_bot_safe_crm_context(...).found=true` и
`timeline_context.bot_context.items>0`. AMO note_id: `471858683`,
`471858687`, `471858693`, `471858695`, `471858701`; по ним
`memory_hits`: `1`, `1`, `1`, `7`, `1`. Один кейс содержит email + telegram
memory (`mail_archive_stage2:email_message` ×6,
`telegram_history:channel_message` ×1), остальные — `bot_safe_summary`.

Границы: `DRAFT_LOOP_AUTO_RESOLVER=0`, основные Wappi pairs/profiles не
менялись, клиентам `0` отправок (`client_sends_delta=0`), AMO write только
draft notes с маркером `ЧЕРНОВИК БОТА, не отправлено`. Ограничение: строгих
свежих чатов, где последний Wappi message inbound и runtime-память >0, в
снимке не найдено; 4/5 записей второй пятёрки — ретро-проверка памяти по
чатам, на которые менеджер уже отвечал позже. Семантический статус:
`PASS_WITH_NOTES`.

Операционный риск `getUpdates Conflict` проверен: локально найден только один
public Telegram poller — PID `42671` в screen
`mango_public_pilot_bots_main_a23dede6_clean_20260707`, heartbeat свежий
(`status=polling`, `effective_profile=pilot_gold_v1`). Второй локальный
poller не найден, поэтому рабочий live-процесс не гасился. Yandex/OpenClaw
rollback backup повторно подтверждён sha256
`ef9ef249b4192b768cd1eb826f6df20514994539a3911f9aeee19bbc295d03c8`.

### D-072. Nightly warehouse now sweeps processed Mango call folders before import

Решение владельца от 2026-07-07: исправить именно штатный ночной процесс, а не
разово подставлять файл со звонками. Добавлен обязательный шаг
`mango_processed_sweep` перед `calls_and_amo_incremental` в Dv2 nightly chain.

Шаг сканирует локальные `product_data/mango_update_after_*`, берёт только уже
обработанные `call_records` с `analysis_status='done'`, не запускает ASR,
Resolve+Analyze, LLM и сетевой Mango API, строит JSONL в формате существующего
`build_mango_call_timeline_increment.py` и кладёт его в
`.codex_local/staging/nightly_dv2_sources/`. Далее штатный
`MangoCallSummaryNormalizer` импортирует JSONL в staging Customer Timeline.

Критерий публикации следующего prod snapshot по звонкам: реальный SQL-прирост
`timeline_events.source_system='mango_processed_summary'` и продвижение
`MAX(event_at)`, а не общий `changed_customer_count`. Известное поведение:
AMO contact snapshot может давать повторный `updated` в 5-минутном overlap,
поэтому rerun=0 для звонков проверяется отдельно по `mango_processed_summary`.

### D-073. Snapshot #2 published after processed Mango call sweep

Решение владельца от 2026-07-07 исполнено: после `mango_processed_sweep`
опубликован prod snapshot #2 штатным инструментом `scripts/publish_snapshot/`.
Снимок:
`product_data/customer_timeline/prod_snapshots/prod_20260707_211300_calls_sweep/customer_timeline.sqlite`,
sha256 `eb38dc7a8790f55cbc31d28381f420403a7bcdc3af460ac00aff66e965c1e0e9`.

Фактический прирост от sweep: `timeline_events` по
`source_system='mango_processed_summary'` выросли с `74029` до `75027`,
`MAX(event_at)` продвинулся с `2026-07-01T15:03:25+00:00` до
`2026-07-07T13:44:45+00:00`; `bot_context_chunks` по тому же источнику выросли
до `72765`. Повторный nightly run дал `0` SQL-прироста по
`mango_processed_summary`; один overlap/update в AMO safety-window не считается
дублем звонков.

Публикация: staging `wal_checkpoint(TRUNCATE)` вернул `[0,0,0]`, staging WAL
обнулён, `reader_smoke` по 5 контрольным клиентам прошёл, prod после flip:
`timeline_events=175586`, `bot_context_chunks=133428`, `quick_check=ok`.
Rollback backup создан и sha-проверен:
`product_data/customer_timeline/prod_backups/pre_flip_backup_2026-07-07T211813.897805Z0000/customer_timeline.sqlite`;
асинхронная копия создана в `~/Yandex.Disk.localized/OpenClaw/prod_backups/`.

Операционное уточнение: live-worktree уже был на
`15accd2ebabf7007f62ae6dafe04bad4548c91ba`, а старый launcher был жёстко
закреплён на `a23dede6`, поэтому старт после flip был переведён на новый
эквивалентный launcher под текущий HEAD и запускается через `screen`
`mango_public_pilot_bots_main_15accd2_snapshot2_20260708`. Клиентам сообщений
не отправлялось; e2e заменён на non-send проверки heartbeat, process/screen,
sha/counts/reader-smoke.

### D-074. Mail-merge snapshot #3 publishes through Wappi draft-loop reader only

Решение владельца от 2026-07-09: после M1 mail merge публичный Telegram-бот
остается выключенным, а единственный прикладной читатель стабильного Customer
Timeline prod-пути для публикации snapshot #3 — Wappi→AMO draft-loop.

Конфиг `scripts/publish_snapshot/config.marathon2_noch_current.json` переведен
с owner-gated заглушек на реальные команды остановки/старта
`run_amo_wappi_draft_loop.py`: start через screen `mango_draft_loop`, память
в prompt включена (`TELEGRAM_TIMELINE_MEMORY_IN_PROMPT=1`,
`TELEGRAM_BOT_SAFE_CRM_CONTEXT=1`), auto-resolver выключен
(`DRAFT_LOOP_AUTO_RESOLVER=0`), auto-pairs для постоянного процесса заменены
на локальный пустой файл, модель/reasoning — `gpt-5.5/high`.

Граница безопасности: клиентам ничего не отправлять; разрешённая внешняя запись
для draft-loop — только AMO manager note draft через AI Office endpoint с
маркером `ЧЕРНОВИК БОТА, не отправлено`. Watchdog draft-loop на время flip
останавливается, чтобы не держать prod DB; после успешного smoke возвращается.

### D-075. Mail-merge snapshot #3 rolled back on mail safety gate

Исполнение 2026-07-09: snapshot
`prod_snapshots/prod_20260709_m1_mail_merge_e3cd3fb7/customer_timeline.sqlite`
был опубликован на стабильный prod-путь и прошёл формальный `reader_smoke`
по 5 контрольным клиентам. Новый prod sha:
`06f4c081f4336280dac95c8b95ec53d042b1be9eb979510a80218fa86ed0a5a3`;
rollback backup старого prod sha:
`eb38dc7a8790f55cbc31d28381f420403a7bcdc3af460ac00aff66e965c1e0e9`.

После независимого аудита найден post-gate blocker: в опубликованном snapshot
`7919` opened mail chunks (`mail_archive_stage2`, `allowed_for_bot=1`) связаны
с A2 facts, где есть `manager_action_required`/`has_manager_note`, и `7921`
opened mail chunks имеют `client_safe=0`. Это нарушает правило публикации:
почтовые chunks, противоречащие A2 facts `client_safe`/`bot_visible`/manager
tags, не должны быть открыты боту.

Решение: Wappi draft-loop и watchdog остановлены, snapshot #3 откатан через
`scripts/publish_snapshot/rollback.py` на sha-проверенный backup. Prod после
rollback: sha `eb38dc7a8790f55cbc31d28381f420403a7bcdc3af460ac00aff66e965c1e0e9`,
`quick_check=ok`, `email_summary_cache_v1=1507`. Клиентам сообщений не
отправлялось. AMO note writes в этом SWAP-заходе не выполнялись.

Постоянная защита: `reader_smoke.py` теперь включает `mail_allowed_safety_gate`
и падает, если opened mail chunks имеют связанные A2 facts с
`client_safe=0`, `bot_visible=0`, `manager_action_required` или
`has_manager_note`. Следующий SWAP разрешён только после исправления
stage4b/mail opening на staging и зелёного `mail_allowed_safety_gate=ok`.

### D-076. Mail opening Variant B is constrained by A2 bot_visible

Решение владельца/архитектора от 2026-07-09: для Wappi→AMO manager drafts
разрешён Вариант Б — письма с денежными/налоговыми/договорными фактами могут
быть открыты в память черновика, если A2 явно пометил их `bot_visible=1`.
Это не означает автономную отправку клиенту: клиентам сообщений не отправлять,
финальный текст видит и отправляет менеджер.

Исполнительное правило: `mail_archive_stage2` chunk можно открыть
(`allowed_for_bot=1`, `requires_manager_review=0`) только если есть связанная
строка `a2v3_mail_event_facts.bot_visible=1` и нет секретных тегов
`sensitive_credentials`/`sensitive_bank_requisites`/
`sensitive_payment_details`/`sensitive_personal_data`/
`sensitive_document_data`/`sensitive_medical`. Деньги/налоги/договоры
(`sensitive_money`, `sensitive_tax`, `sensitive_contract`) разрешены только как
manager-draft context при `bot_visible=1`.

Защита публикации: `reader_smoke.mail_allowed_safety_gate` обязан падать на
открытом письме без A2-факта, с `bot_visible=0`, с секретным тегом или с
первичной причиной `manager_action_required`/`has_manager_note`; Variant B
считается отдельно и не является нарушением.
