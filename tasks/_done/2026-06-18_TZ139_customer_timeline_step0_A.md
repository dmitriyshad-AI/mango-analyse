> DONE 2026-06-18 13:07 | ветка codex/tz139-customer-timeline | codex

> TAKE 2026-06-18 12:34 | ветка codex/tz139-customer-timeline | codex

Ветка: codex/tz139-customer-timeline
Зоны: src/mango_mvp/customer_timeline/, tests/, tasks/, docs/worktrees_registry.md, audits/_inbox/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_customer_timeline_approval_decisions.py tests/test_customer_timeline_approval_workspace.py tests/test_customer_timeline_approved_context_pack.py tests/test_customer_timeline_canonical_readonly_import.py tests/test_customer_timeline_canonical_readonly_triage.py tests/test_customer_timeline_channel_preview_from_pack.py tests/test_customer_timeline_contact_control_sample_import.py tests/test_customer_timeline_context_provider.py tests/test_customer_timeline_contracts.py tests/test_customer_timeline_deal_aware_sample_import.py tests/test_customer_timeline_import_cli.py tests/test_customer_timeline_ingestion.py tests/test_customer_timeline_preview_quality_audit.py tests/test_customer_timeline_read_api.py tests/test_customer_timeline_store.py
Семантический-аудит: да

# TZ139 customer_timeline: Step 0 + Work A

Источник задания: вложение Дмитрия от Claude `2026-06-17_TZ139_svodnaya_istoriya_klienta...` / audited execution plan.

## Жёсткие границы

- Работать только в отдельном worktree `codex/tz139-customer-timeline`.
- Не писать в AMO, Tallanto, CRM и внешние системы.
- Не запускать ASR, Resolve+Analyze, тяжёлые batch/start/run-ui скрипты.
- Источники читать только read-only. Source DB открывать через `mode=ro`.
- Не менять `stable_runtime/`, `runs/`, `transcripts/`, исходные source DB.
- Без `git reset`, `git checkout`, `git clean`, удаления файлов.
- После Work A остановиться, отчитаться и ждать ревью перед Work B.

## Step 0: read-only baseline

Подтвердить по текущему коду:

- `read_api.py`: `safe_for_automatic_bot`.
- `ids.py`: `stable_customer_id`, `stable_signal_id`, отсутствие metadata в signal id.
- `ingestion.py`: `allowed_for_bot=False` для импортируемых chunks.
- `canonical_readonly_import.py` и `canonical_readonly_triage.py`: текущий `no_mango_calls`.
- `store.py`: scrub / `FORBIDDEN_PERSISTED_PAYLOAD_KEYS`, WAL/busy_timeout, `record_conflict`, `ingestion_runs`.

Запустить зелёный baseline по customer_timeline тестам из шапки ТЗ и safety contract.

## Work A: entity resolution + brand model

Цель: подготовить customer_timeline к реальной клиентской карточке без тихих склеек клиентов.

Требования:

- Телефон как ключ связи разрешён для одного человека: общий телефон объединяет события одного клиента.
- Семейный телефон с несколькими учениками Tallanto не склеивает разных детей в одного клиента.
- Конфликтные/неоднозначные связи фиксируются явно, а не мержатся молча.
- Добавить таблицу соответствия старый `customer_id` -> новый `customer_id`; mapping должен быть полным и обратимым.
- Бренд хранить как многозначный атрибут истории клиента; не использовать бренд как блокер merge.
- Foton/UNPK не смешивать в клиентском/менеджерском тексте: разделение на выводе будет в следующих работах, не в Work A.
- Сохранить safety-инварианты: raw payload не сохраняется, bot-safe не расширяется, live write отсутствует.

## Приёмка Work A

- Есть NEG-тесты на отсутствие тихой склейки семейного телефона.
- Есть тест на полный/обратимый old->new customer_id mapping.
- Есть тест, что brand не блокирует идентичность, но сохраняется как многозначная история.
- Модульные customer_timeline тесты зелёные.
- `assert_customer_timeline_safety_contract` проходит после изменений.
- Создан audit pack с формальным и смысловым review.
- Отдельный коммит только по Step 0 + Work A.
