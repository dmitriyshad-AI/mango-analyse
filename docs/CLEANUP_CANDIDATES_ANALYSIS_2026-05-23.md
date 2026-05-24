# Cleanup candidates analysis - 2026-05-23

## Scope

Анализ выполнен без удаления и без переноса файлов.

Проверялись:

- `stable_runtime/CURRENT_RUNTIME.json`;
- `stable_runtime/CANONICAL_EXPORT.txt`;
- текущий audio working store;
- крупные папки верхнего уровня;
- крупные DB/CSV/XLSX/ZIP;
- ссылки в `scripts/`, `src/`, `tests/`, `docs/`.

## Что защищено и не предлагается удалять сейчас

| Путь | Размер | Почему оставить |
|---|---:|---|
| `_external_handoffs/mail_archive_2026-05-12` | 65G | По текущим решениям почту пока не удаляем; это будущий источник единой истории клиента. Можно вынести из проекта во внешний архив, но не удалять. |
| `product_data/audio_working_store_20260523_v1` | 25G | Единственная рабочая папка Mango-аудио. |
| `stable_runtime/canonical_master_20260523_audio_working_store_v1` | 1.5G | Текущая canonical DB. |
| `stable_runtime/sales_master_export_20260523_audio_working_store_v1` | текущий export | Активный export по `CANONICAL_EXPORT.txt`. |
| `stable_runtime/crm_writeback_quality_gate_20260523_audio_working_store_v1` | small | Текущий CRM quality gate. |
| `stable_runtime/amo_writeback_queue_20260523_audio_working_store_v1` | small | Текущая AMO queue summary. |
| `product_data/customer_timeline/canonical_readonly_20260521_v5` | 1.4G | Текущий read-only слой customer timeline. Пока не включён как primary, но является актуальным экспериментальным источником. |
| `_local_archive_mango_api_downloads_20260507/product_appliance` | 35M root total | На него ссылается `CURRENT_RUNTIME.json` как product appliance; оставлять. |
| `product_data/knowledge_base/kb_release_20260520_v6_3_team_answers*` | 44M total | Текущая база знаний и bot/employee packs. |
| `.venv-asrbench` | 1.0G | Рабочее ASR-окружение; можно удалить только если есть план пересоздания. |

## Высокий приоритет cleanup после отдельного подтверждения

Эти элементы не являются текущим runtime и дают заметную экономию/чистоту. Рекомендованный способ: перенос в корзину с manifest, не `rm`.

| Приоритет | Путь | Размер | Риск | Что сделать перед переносом |
|---:|---|---:|---|---|
| 1 | `stable_runtime/canonical_master_20260521_after_mango_update_v1` | 1.5G | Низкий | Текущий runtime уже использует `canonical_master_20260523_audio_working_store_v1`. Проверить `CURRENT_RUNTIME.validation_ok=true`. |
| 2 | `stable_runtime/sales_master_export_20260521_after_mango_update_v4_runtime_acceptance` | 235M | Низкий | Текущий export уже `sales_master_export_20260523_audio_working_store_v1`. |
| 3 | `product_data/canonical_audio_store_20260516_v1` | 349M | Низкий-средний | Старый audio store заменён. Audio уже вынесен, остались manifests/projection. Можно удалить после сохранения текущего cleanup manifest. |
| 4 | `_local_archive_20260424` | 352M | Низкий | Старый локальный архив/test DB/zip. Не участвует в текущем runtime. |
| 5 | `stable_runtime/ra_pending_mango_api_20260517_v1` | 52M | Низкий | Текущий runtime: missing R+A = 0. Это старый pending слой. |
| 6 | `stable_runtime/history_remaining_excl_done_20260407` | 85M | Низкий | Историческая старая DB, не текущий источник. |

Примерная экономия первого безопасного batch: около `2.6G`.

## Cleanup после небольшой правки тестов/скриптов

Эти элементы можно убрать, но сначала надо заменить старые прямые ссылки на current-runtime или маленькие fixture-файлы.

| Путь | Размер | Блокер |
|---|---:|---|
| `stable_runtime/deal_aware_stage2_attribution_20260514_selector_fix_phase1` | 316M | `tests/test_deal_aware_confidence_recalibration.py` читает этот слой как baseline. Нужно заменить на fixture/summary. |
| `stable_runtime/deal_aware_stage2_attribution_20260514_selector_fix_phase2` | 318M | Нужен только как исторический Phase2 artifact; можно сжать до summary/CSV samples после проверки downstream. |
| `stable_runtime/deal_aware_stage3_deal_state_20260514_selector_fix_phase2` | 407M | Нужен только для исторического deal-aware selector fix. Можно оставить summary/report, убрать `.sqlite` и тяжёлые CSV после manifest. |
| `stable_runtime/sales_master_export_20260513_human_history_v8_normalized` | 294M | `scripts/build_deal_aware_stage1_snapshot.py` hardcoded на этот export. Нужно перевести скрипт на current-runtime или пометить legacy. |
| `product_data/mango_audio_update_20260516_v1` | 77M | `scripts/build_audio_store_downstream_projection.py` ещё использует `asr_handoff_new_calls_20260516.csv` как default queue. Нужно заменить default на актуальный пустой/controlled queue или сделать аргумент обязательным. |
| `product_data/mango_update_after_20260512_20260521_v1` | 89M | `docs/ASR_RUNTIME_CONTRACT_2026-05-21.md` и `scripts/check_asr_runtime_contract.py` ссылаются на старый batch. Нужно обновить contract или пометить batch closed. |

Примерная экономия после правок: около `1.5G`.

## Большие, но лучше не удалять, а вынести из проекта

| Путь | Размер | Рекомендация |
|---|---:|---|
| `_external_handoffs/mail_archive_2026-05-12` | 65G | Не удалять. Лучше перенести за пределы repo, например в `~/Foton_Data_Archive/mail_archive_2026-05-12`, и оставить README/pointer + runbook. |
| `telegram_exports (2)` | 1.2G | Содержит сырой Telegram export и media. Используется как исторический источник для question catalog/customer timeline. Лучше вынести во внешний read-only архив и оставить pointer. |
| `TP UNPK DataExport_2026-05-21` | 159M | Telegram data export, не Mango. Лучше оставить до завершения Telegram/history этапа или вынести во внешний архив. |

## Локальные кэши, которые можно чистить отдельно

| Путь | Размер | Риск |
|---|---:|---|
| `.codex_local/question_catalog` | 115M | Низкий, если текущие product_data outputs сохранены. |
| `.codex_local/kc_source_extract_20260513` | 89M | Низкий-средний; старый extract. |
| `.codex_local/google_docs_lvsh_review` | 81M | Низкий-средний; review cache. |
| `.codex_local/llm_review_home` | 34M | Низкий; кэш/рабочие файлы. |
| `.cache/llm_responses` | 7.6M | Низкий; кэш. |

Не удалять `.codex_local/auth.json`, `.codex_local/config.toml`, `.codex_local/skills`, `.codex_local/plugins` без отдельной причины.

## Root-файлы и Excel

Небольшие, но засоряют корень. Кандидаты на перенос в `external_exports_archive` или корзину после manifest:

- `АКТУАЛЬНО_*.xlsx` - многие уже stale по текущему runtime;
- `Contacts.xls` - старый Tallanto/contact input, если normalized snapshots сохранены;
- `260512_*write_off_visits_from_class.xlsx` - исходные Tallanto reports, лучше оставить до полного Tallanto-abonement этапа или вынести во внешний архив;
- `external_m1_jan_mar_2025_asr_only_20260504_result_20260506_103021.zip` - старый ASR zip;
- `Mango_Bot_KB_FINAL_v6_3_2026-05-20` - если это дубль текущего KB release, можно убрать после сравнения.

## Что не считать мусором автоматически

- `mango_mvp.db` и `mango_mvp.db-shm`: старый root runtime DB, но часть legacy GUI/default config всё ещё ссылается на `sqlite:///mango_mvp.db`. Удалять только после проверки процессов и решения, что legacy GUI/worker больше не используются.
- `ai_office.db`: default DB для AMO runtime config; маленький, лучше оставить.
- `product_data/customer_timeline/canonical_readonly_20260521_v5`: большой, но актуальный слой для следующего customer timeline этапа.
- `stable_runtime/amocrm_runtime`: содержит AMO runtime/writeback отчеты; оставить.

## Рекомендованный следующий cleanup batch

1. Сделать manifest `docs/RUNTIME_CLEANUP_BATCH3_CANDIDATES_2026-05-23.md`.
2. Перенести в корзину high-confidence batch:
   - `stable_runtime/canonical_master_20260521_after_mango_update_v1`;
   - `stable_runtime/sales_master_export_20260521_after_mango_update_v4_runtime_acceptance`;
   - `product_data/canonical_audio_store_20260516_v1`;
   - `_local_archive_20260424`;
   - `stable_runtime/ra_pending_mango_api_20260517_v1`;
   - `stable_runtime/history_remaining_excl_done_20260407`.
3. Проверить:
   - `CURRENT_RUNTIME.validation_ok=true`;
   - `pytest --collect-only -q`;
   - audio projection smoke;
   - targeted productization/runtime tests.
4. Отдельным engineering cleanup заменить прямые ссылки в тестах/скриптах на старые deal-aware/export папки.
5. Только после этого чистить старые deal-aware stage DB/CSV и Mango update batch-пакеты.

## Оценка

Без почты можно безопасно убрать примерно `2.6G` уже следующим batch и ещё около `1.5G` после небольших правок тестов/скриптов.

Если цель - именно радикально освободить диск, главный рычаг не runtime cleanup, а перенос `_external_handoffs/mail_archive_2026-05-12` за пределы repo: `65G`. Но удалять его сейчас не рекомендую.
