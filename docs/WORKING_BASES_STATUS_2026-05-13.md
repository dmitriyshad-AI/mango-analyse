# Статус рабочих баз и runtime-слоев на 2026-05-13

## Короткий вывод

База распознанных и обработанных звонков актуальна: все actionable-аудио за период 2025-01-01 — 2026-05-31 имеют ASR и полный Resolve+Analyze. Проблема не в распознавании и не в R+A, а в downstream-слоях: часть рабочих CSV/XLSX и AMO-ready пакетов была собрана до доработки человекочитаемой истории и deal-aware логики.

## Источник истины по звонкам

Актуальная canonical-база:

`stable_runtime/canonical_master_20260510_after_quality_backfill_v1/canonical_calls_master.db`

Актуальный call-level слой для истории по телефонам:

`stable_runtime/insight_readiness_report_after_quality_backfill_20260510_v1/calls_terminal_analyzed.csv`

Контрольные числа:

- actionable source audio: 64 832
- ASR done: 64 832
- Resolve+Analyze done: 64 832
- missing ASR: 0
- missing R+A: 0
- calls with phone: 63 788
- calls without phone: 1 044
- unique client phones: 15 924
- contentful calls: 46 153 в readiness-отчете; 43 456 в текущем AMO-export builder после дополнительного CRM low-value фильтра

## Что было устаревшим

`client_chains.csv` остается полезным аналитическим слоем по телефону, но его поля `products_top`, `objections_top`, `example_latest_summary` слишком агрегатные для менеджера. Они давали шум вроде `летний лагерь | летний лагерь: 14` и слишком короткую хронологию.

Старая логика была приемлема для технического contact-layer, но не для карточки сотрудника и не для deal-aware CRM UX.

## Что обновлено сейчас

Обновлены скрипты:

- `scripts/build_post_backfill_amo_ready_export.py`
- `scripts/build_student_card_manual_review_pack.py`
- `scripts/write_amo_ready_contacts.py`
- `src/mango_mvp/quality/crm_text_quality_detector.py`

Новая логика:

- `Краткая история общения` = короткая управленческая сводка по клиенту, без счетчиков и длинного дубля последнего звонка.
- `Хронология общения (последние 5 касаний)` технически оставлена под старым названием для совместимости, но теперь содержит все содержательные звонки по телефону, каждый нормальным предложением.
- `Авто история общения` при записи в AMO больше не вставляет полную хронологию внутрь карточки, чтобы не заставлять сотрудника читать одно и то же несколько раз.
- Проверка `cross_field_duplicate_information` больше не считает рабочую хронологию writeback-полем, потому что она не пишется в AMO как отдельное поле.
- Добавлен tenant-normalizer: `МПК/НПК/ОМПК/ВНПК/МНПК МФТИ` нормализуются в `УНПК МФТИ`; ASR-ошибка `летние ночные школы` нормализуется в `летние очные школы`; продуктовые счетчики и синонимы схлопываются.

Новый review-pack для РОП:

`stable_runtime/student_card_manual_review_next50_20260513_v5/student_card_manual_review_next50_for_rop.xlsx`

Новый пересобранный AMO/export слой для проверки новой истории:

`stable_runtime/sales_master_export_20260513_human_history_v7/`

## Почему новый human-history export пока не переключен как active runtime

`stable_runtime/sales_master_export_20260513_human_history_v7/amo_export_ready_ru.csv` содержит 54 потенциально готовые строки, но новый CRM quality gate `stable_runtime/crm_writeback_quality_gate_20260513_human_history_v3/summary.json` заблокировал 46 из них по причине `stale_source_next_step`: последние звонки старые относительно 2026-05-13, а следующий шаг был рассчитан из исторического звонка.

Это правильное поведение. Нельзя автоматически писать в CRM старые рекомендации без учета текущих сделок, оплат, задач AMO и статуса клиента.

Поэтому активный указатель пока оставлен на предыдущем строгом слое:

`stable_runtime/CANONICAL_EXPORT.txt -> sales_master_export_20260510_after_quality_backfill_v5_crm_text_quality_strict`

Этот слой не старый апрельский полуфабрикат; он собран из актуальной canonical-базы, но его текстовая история менее человекочитаемая, чем новая v7.

## Что считать устаревшими полуфабрикатами

Не использовать для новых решений без отдельного аудита:

- старые `sales_master_export_20260424...`
- Stage15/ROP/KB артефакты до `v11_frozen_gate`
- AMO-ready CSV/XLSX, собранные до `sales_master_export_20260510_after_quality_backfill_v5_crm_text_quality_strict`
- любые временные пакеты `*_preview_*`, `*_dryrun_*`, `*_audit_*`, если они не указаны в `stable_runtime/CURRENT_RUNTIME.json` или в конкретном плане текущего этапа

## Что делать дальше

Следующий правильный шаг не просто переключить pointer, а собрать новый deal-aware слой:

1. Брать за основу актуальные calls_terminal/canonical данные и новую человекочитаемую историю.
2. Подтягивать текущие AMO-сделки, статусы, причины отказа, задачи, оплаты и Tallanto-контекст.
3. Разделять контактную сводку и сделочную сводку.
4. Пересчитывать следующий шаг и вероятность продажи с учетом текущего статуса сделки, а не только последнего звонка.
5. После этого запускать CRM quality gate, Claude audit и только затем staged AMO writeback/repair.
