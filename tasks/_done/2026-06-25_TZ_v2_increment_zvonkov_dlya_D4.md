> DONE 2026-06-25 17:27 | ветка codex/mango-call-increment | codex

> TAKE 2026-06-25 17:01 | ветка codex/mango-call-increment | codex

Ветка: codex/mango-call-increment
Зоны: scripts/, src/mango_mvp/customer_timeline/, tests/, tasks/_running/, tasks/_done/, audits/_inbox/, docs/worktrees_registry.md
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_customer_timeline_contracts.py tests/test_customer_timeline_store.py tests/test_customer_timeline_ingestion.py tests/test_customer_timeline_nightly_incremental.py tests/test_customer_timeline_canonical_readonly_import.py tests/test_productization_call_processing_readiness.py tests/test_tz19_calls_review_table.py
Семантический-аудит: да

# ТЗ v2 (для D4) — инкремент звонков Mango → Customer Timeline

Дата: 2026-06-25. v2 после регрейда D4 — внесены 7 правок (все блокеры D4 подтверждены Claude #1 по сырью). v1 содержал реальные ошибки (source_system, создание клиента из воздуха, расположение июньских звонков, имя колонки телефона) — исправлено.

## Цель (без изменений)
Тонкий безопасный инкремент: новые обработанные звонки Mango → Customer Timeline по курсору, без дублей и неверной привязки, на ТЕСТОВОЙ копии, без переписывания готового конвейера, без ASR/Analyze в рамках этого ТЗ.

## База реализации (правка #1)
Отдельный clean worktree от **`4caa5eb` / `codex/release-venue-autonomy`** (там курсорный раннер `nightly_incremental.py` + таблица `ingestion_cursors`). В `tz135` и `origin/main` этих файлов НЕТ — не делать там. Сверить перед стартом: наличие `run_customer_timeline_nightly_incremental.py`, `ingestion_cursors`, `MangoCallSummaryNormalizer`.

## Источник данных v1 (правки #5, #6)
- Брать ТОЛЬКО уже обработанные звонки: `analysis_status='done'` + валидный непустой `analysis_json`. **ASR/Analyze/Mango-download в этом ТЗ НЕ запускать** — отдельный шаг с отдельным «да».
- Читать ДВА источника готового разбора:
  1. `canonical_calls` (canonical DB, до ~2026-05-21);
  2. package-local `call_records` в `product_data/mango_update_after_*` (свежие, в т.ч. июнь — их в canonical НЕТ).
  Если по решению владельца ограничиваемся только canonical — написать это явно и НЕ трогать пакеты (но тогда июнь не довнесём).
- Телефон: колонка **`phone`** (правка #7; принимать алиасы `client_phone/normalized_phone/«Телефон клиента»`, но основная — `phone`).

## source_system и дедуп (правка #2 — критично от дублей)
- Использовать СУЩЕСТВУЮЩИЙ `source_system="mango_processed_summary"` (event_type `mango_call`) — как уже лежат 71 962 чанка в боевой. **НЕ заводить новый `source_system="mango_call"`** — это параллельная история + риск дублей (dedupe_key включает source_system).
- Дедуп — существующий `dedupe_key` (UNIQUE-индекс store.py). Повторный прогон = 0 новых событий.

## Привязка к клиенту — продюсер резолвит САМ, без создания из воздуха (правка #3 — критично от неверной привязки)
⚠ Существующий `MangoCallSummaryNormalizer` (ingestion.py ~851) при наличии телефона СОЗДАЁТ клиента с `match_class=STRONG_UNIQUE` без проверки на семейный/общий телефон (проверено по сырью, строки ~10-93). Безопасная ambiguous-логика есть только в полном `canonical_readonly_import`. Поэтому:
- **Продюсер делает identity-резолв САМ против существующих `identity_links` боевой timeline** (через read-only): телефон → ровно один customer = `strong_unique`; несколько/общий/семейный = `ambiguous`; нет совпадения = `unmatched`.
- Продюсер пишет в JSONL уже **resolved-поля**: `customer_id` (если strong_unique) + `match_class`. Нормализатор НЕ должен создавать нового клиента: `ambiguous`/`unmatched` → событие в карантин/pending, НЕ приписывать одному клиенту и НЕ плодить customer из воздуха.
- Семейный/общий телефон → `ambiguous`, ручная проверка, не один ребёнок.

## Раннер и пересборка затронутых (правка #4)
- Раннер `run_customer_timeline_nightly_incremental.py` должен выбирать ОДИН И ТОТ ЖЕ Mango-нормализатор и для загрузки (`load_incremental_jsonl_source` → расчёт `changed_customer_ids`), и для импорта. Сейчас по умолчанию там `JsonlTimelineNormalizer`; если переключить только в одном месте, пересборка затронутых будет пустой/неверной — переключить согласованно.
- Пересборка bot_safe_summary только затронутых (`changed_customer_ids`) — в v1 НЕ собираем (финальный сборщик `deferred`), но `changed_customer_ids` должны считаться корректно для будущего.

## non_conversation (правка)
Сервисные недозвоны/`non_conversation` хранить как событие БЕЗ содержательной истории и без полезного chunk (не попадают в карточку/память). Полезные типы (`sales_call/existing_client_progress/technical`) сохраняются.

## Безопасность БД (без изменений, усилено)
- Боевую timeline НЕ трогать. Работать на ТЕСТОВОЙ копии, снятой ТОЛЬКО через SQLite `.backup` (готовый `create_backup` в `scripts/repair_mail_stage2_event_dates.py:177`); `cp`/`rsync` боевой WAL-БД ЗАПРЕЩЕНЫ.
- Не писать в AMO/Tallanto/CRM. Не менять stable_runtime. Новые артефакты — в датированные папки. Никаких git reset/clean.

## Первый контрольный прогон v1
- Окно: 1 день свежих звонков ИЛИ ≤20 звонков, только `analysis_status='done'`.
- Тестовая копия timeline через `.backup`; датированная рабочая папка.
- ASR/Analyze НЕ запускать (использовать готовый `analysis_json`).

## Проверки (NEG)
- Повтор того же окна: новых событий = 0 (дедуп).
- Граница курсора (overlap 300с): события не пропадают.
- Семейный/общий телефон → `ambiguous`, не один ребёнок; нет совпадения → `unmatched`; клиента из воздуха НЕ создаём.
- Сырые разборы: `allowed_for_bot=0`, `requires_manager_review=1`.
- `non_conversation` не в содержательную историю.
- `changed_customer_ids` считается верно (не пустой при реальных новых, не «чужой»).
- Mango/источник недоступен или сбой → курсор НЕ двигается, не портится.
- source_system = `mango_processed_summary` (не появился параллельный `mango_call`).

## Отчёт Claude #1 (tasks/_done + audit pack)
Найдено/новых/из канона/из пакетов; linked(strong_unique)/ambiguous/unmatched; событий добавлено; repeat=0 дублей; время фаз; путь тестовой копии + рабочей папки; подтверждение «source_system=mango_processed_summary, клиент из воздуха не создан, ambiguous в карантин»; `changed_customer_ids` корректны; `allowed_for_bot=0`; 10 примеров с исходом привязки. Вердикт «в прод» НЕ выносить.

## Стоп-условия
Требуется ASR/Analyze/download без «да»; запись в боевую timeline/AMO/Tallanto; копия не через `.backup`; продюсер создаёт клиента из воздуха или приписывает семейный телефон одному; появился параллельный source_system `mango_call`; массовый ambiguous.
