> DONE 2026-06-25 21:00 | ветка codex/phase1-dossier-enrich | codex

> TAKE 2026-06-25 20:38 | ветка codex/phase1-dossier-enrich | codex

Ветка: codex/phase1-dossier-enrich
Зоны: src/mango_mvp/customer_timeline/bot_safe_summary.py, src/mango_mvp/customer_timeline/bot_safe_runtime_context.py, src/mango_mvp/channels/subscription_llm_parts/direct_path.py, tests/, tasks/_done/, tasks/_running/, docs/worktrees_registry.md, audits/_inbox/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_customer_timeline_approval_decisions.py tests/test_customer_timeline_context_provider.py tests/test_customer_timeline_deal_aware_sample_import.py tests/test_customer_timeline_contracts.py tests/test_customer_timeline_canonical_readonly_import.py tests/test_customer_timeline_approved_context_pack.py tests/test_customer_timeline_approval_workspace.py tests/test_customer_timeline_store.py tests/test_bot_safe_runtime_context.py tests/test_customer_timeline_next_step_resolver.py tests/test_customer_timeline_derived_signals.py tests/test_bot_safe_direct_path_context.py tests/test_direct_path_known_slots_next_step_prompt.py tests/test_customer_timeline_full_memory_ingest.py tests/test_customer_timeline_mail_stage2_ingest.py tests/test_customer_timeline_channel_preview_from_pack.py tests/test_customer_timeline_import_cli.py tests/test_customer_timeline_canonical_readonly_triage.py tests/test_customer_timeline_ingestion.py tests/test_customer_timeline_preview_quality_audit.py tests/test_customer_timeline_bot_safe_summary.py tests/test_customer_timeline_nightly_incremental.py tests/test_customer_timeline_read_api.py tests/test_customer_timeline_contact_control_sample_import.py tests/test_bot_safe_memory_step_guard.py
Семантический-аудит: да

# ТЗ Фаза 1 (Кодекс / Д3) — обогащение bot-safe ДОСЬЕ из звонков
Производная от мастер-ТЗ `2026-06-25_TZ_bot_full_client_history_context.md` (фазовый план, заземлён в коде). Это Фаза 1 = минимальный путь до первой пользы. Память остаётся за флагом, дефолт OFF.

## Цель
Заменить пустой 4-польный шаблон выжимки на СТРУКТУРИРОВАННОЕ безопасное досье, наполненное из УЖЕ имеющихся богатых саммари звонков. **«Звонки» = `timeline_events` с `event_type='mango_call'` И `source_system='mango_processed_summary'`** (так они лежат в боевой; НЕ создавать `source_system='mango_call'` — это запрещённый источник дублей). До 1348 знаков, уже в таймлайне. Без новых источников, без identity-переделок.

## Якоря в коде (проверены по сырью)
- `customer_timeline/bot_safe_summary.py`: `_render_safe_text` (314-320) — текущий 4-польный шаблон («Бренд/Стадия/Интерес/Следующий шаг»); `_build_customer_brand_drafts`/`_build_customer_draft` (~246/289) — сборка per-brand; `_scrub_interest_person_names`; NULL-фильтр (507). Сборку per-brand и NULL-фильтр НЕ трогать (держат бренд и пул NULL).
- `channels/subscription_llm_parts/direct_path.py`: редактор «скрыть точную деталь» `_BOT_SAFE_MEMORY_EXACT_DETAIL_RE` (99); `_direct_path_bot_safe_context_items` (396); `UNSAFE_INTEREST_MARKERS`; флаг `TELEGRAM_BOT_SAFE_CRM_CONTEXT` (89); рантайм-лимиты `_truncate` (~700/1800).

## Scope
1. Перестроить досье на МНОГОСТРОЧНОЕ: блоки «обсуждали», «интерес/возражения», «договорённость / следующий шаг (со `status`)». Источник — саммари звонков из таймлайна. Шапка (стадия/интерес/следующий шаг) остаётся СВЕРХУ как оглавление; под ней — содержательные строки.
2. Поднять рантайм-лимиты `_truncate` (700/1800) и `limit` чанков ОСОЗНАННО — размер позволяет (p99 ~7к токенов), но см. п.3.
3. ⚠ КРИТИЧНО (вывод OFF-регрейда 25.06): на автономном маршруте бот уже течёт «производными клеймами» (`derived_product_claim`: «составим, чтоб предметы не пересекались», «чек ускорит подтверждение», «есть несколько вариантов»). Обогащение УСИЛИТ соблазн. Точные числа УЖЕ частично закрыты: `_BOT_SAFE_MEMORY_EXACT_DETAIL_RE` (direct_path.py:99-110) ловит даты/время-диапазоны/цены/проценты — НЕ дублировать, а ПЕРЕПРОВЕРИТЬ и РАСШИРИТЬ на: адреса (улицы), именованное расписание/группы/даты старта, реквизиты/счёт, назначение платежа, точные дедлайны; добавить регрессионные тесты. ⚠ Производные/процедурные клеймы — это СЕМАНТИКА, регексом не ловятся: чинить структурно — досье делать ЭКСТРАКТИВНЫМ (что обсуждали / о чём договорились — фактами), а НЕ генеративным («составим/ускорит/подберём»). Никаких обещаний от имени менеджера со сроком.

## Гейты (не ослаблять)
Бренд (per-brand сборка + рантайм-видимость по активному бренду); NULL-фильтр; неоднозначная личность → пусто; forbidden-assert. Свежесть: «живые» факты (места/оплата/стадия) — не утверждать, оставить «нитью разговора», к менеджеру.
**Имена (из звонков):** скраб частично есть (`_BOT_SAFE_PERSON_CONTEXT_RE` в direct_path, `_scrub_interest_person_names` в bot_safe_summary) — ПЕРЕПРОВЕРИТЬ, что он покрывает имена из САММАРИ ЗВОНКОВ по ВСЕМУ досье (не только поле «интерес»): не выводить имена детей/родителей/преподавателей и любые персональные имена — боту они не нужны. Добавить тесты.
⚠ Person-split (несколько людей под одним `customer_id`) — это Фаза 3, НЕ здесь. В Фазе 1 пер-человек экспозицию НЕ повышать: содержимое берём из звонков, уже привязанных; при неоднозначности личности — пусто (как сейчас).

## Флаг и замер
- Дефолт OFF. Включение — `TELEGRAM_BOT_SAFE_CRM_CONTEXT`.
- Замер на M1 (вручную) на ТОМ ЖЕ наборе, что OFF-реплей (`mango_clean_reliable_replay_20260625`): обогащённое ON vs OFF-опора. Метрика «отправил без правки» (рост над OFF); нули: fabrication ВКЛЮЧАЯ derived-claim, бренд, P0, чужой человек. Не катить, если derived-claim/бренд/P0 выросли.
- Микро-проверка правки в песочнике (10–15 диалогов) ДО полного прогона.

## Границы
Клиенту 0; бренды не смешивать; ПДн/внутр.механики не показывать; боевую timeline на запись не трогать; `stable_runtime` вне флага не менять; секреты не печатать; git reset/clean нельзя.

## Приёмка
Досье многострочное и содержательное; метрику «пустых досье <10%» считать ТОЛЬКО по сегменту клиентов с содержательными звонками (НЕ по всей базе — у части клиентов в timeline нет полезного сырья, это не провал); редактор снимает числа/цены/адреса/расписание/реквизиты, имена не утекают, процедурных клеймов нет (тесты); бренд/NULL/forbidden-гейты целы (тесты зелёные); подготовлен ON-прогон для M1. Вердикт «в прод» НЕ выносить — регрейд Claude #1 по сырью.

## Стоп-условия
Ослаблен forbidden-assert / бренд-гейт; досье подаёт точную деталь или процедурный клейм как факт; пер-человек экспозиция выросла без Фазы 3; запись в боевую БД.
