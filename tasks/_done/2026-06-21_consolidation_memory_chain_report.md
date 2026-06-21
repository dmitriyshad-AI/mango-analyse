# Консолидация проверенной цепочки памяти

Дата: 2026-06-21
Ветка: `codex/consolidation-memory-chain-20260621`
Worktree: `/Users/dmitrijfabarisov/Projects/Mango_consolidation_memory_chain`
Base main: `8ffb752`
Итоговый SHA: см. `git rev-parse --short HEAD` и финальный ответ D1.

## Влито

- `codex/d3-phase01-botsafe-integration` — `b5db248`
- `codex/tz151-known-slots-next-step` — `b5b2566`
- `codex/tz-c-nightly-cursors` — `70e6eb0`
- `codex/memory-judge-grounded` — `125f8c1`

Merge-коммиты:

- `6ef9764` — d3 phase01 botsafe integration
- `22e071e` — known slots next step prompt
- `90f4f6b` — nightly customer timeline cursors
- `c4659ea` — memory grounded judge
- `5e1cd66` — consolidation strict memory OFF under pilot profile
- `185c4a0` — provider no-op NEG for memory guard
- финальный HEAD после отчёта — см. `git rev-parse --short HEAD`.

## Важное расхождение с исходным ТЗ

В `tz151` тест ожидал, что `pilot_gold_v1` включает bot-safe память без `TELEGRAM_BOT_SAFE_CRM_CONTEXT`.
Это противоречило §4 ТЗ: память должна оставаться строго default OFF, а под `pilot_gold_v1` включается только `TELEGRAM_DIRECT_PATH_KNOWN_SLOTS_NEXT_STEP_PROMPT`.

Разрешение:

- `TELEGRAM_BOT_SAFE_CRM_CONTEXT` оставлен строго за `_default_off_flag_enabled`.
- `TELEGRAM_DIRECT_PATH_KNOWN_SLOTS_NEXT_STEP_PROMPT` оставлен ON только под `pilot_gold_v1`.
- Тест `test_pilot_gold_keeps_memory_off_without_extra_context_flag` закрепляет это.
- Добавлен provider-level NEG: при `TELEGRAM_BOT_SAFE_CRM_CONTEXT=0` безусловный вызов `apply_bot_safe_memory_step_guard` не меняет черновик.

## Конфликты и автослияния

- Реальный ручной конфликт: `src/mango_mvp/channels/subscription_llm_parts/direct_path.py`.
- Сохранено вместе:
  - bot-safe блоки D3;
  - PII/person-name фильтр;
  - known-slots / active next_step prompt из tz151;
  - metadata traces.
- Смысловая проверка автослияний:
  - `bot_safe_summary.py`: сохранены `customer_ids`, `retired_stale`, `brand_source_counts`, `next_step_status`, scrub interest/person names.
  - `store.py`: сохранены read-only `as_uri()?mode=ro&immutable=1` и `ingestion_cursors`.
  - `bot_safe_runtime_context.py`: сохранён `next_step_status`.
  - свежие main-файлы `repair_mail_stage2_event_dates.py`, `mail_stage2_ingest.py`, `ingestion.py`, `import_telegram_export_to_timeline.py` не удалены.

## Тесты

Контрактные 9 файлов:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q \
  tests/test_bot_safe_direct_path_context.py \
  tests/test_bot_safe_memory_step_guard.py \
  tests/test_bot_safe_runtime_context.py \
  tests/test_direct_path_known_slots_next_step_prompt.py \
  tests/test_customer_timeline_nightly_incremental.py \
  tests/test_customer_timeline_bot_safe_summary.py \
  tests/test_customer_timeline_store.py \
  tests/test_codex_exec_service_tier.py \
  tests/test_memory_measure_apparatus.py

80 passed in 1.64s
```

После добавления provider NEG:

```text
82 passed in 2.52s
```

Полный pytest:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests
3561 passed, 5 skipped, 1 warning in 82.26s
```

## Проверка тест-файлов и числа тестов

Все 9 контрактных файлов физически присутствуют в `git ls-files`.

Direct-path test counts:

- `tests/test_bot_safe_direct_path_context.py`: source D3 = 8, итог HEAD = 8.
- `tests/test_direct_path_known_slots_next_step_prompt.py`: source tz151 = 10, итог HEAD = 10.

## Флаги

- `TELEGRAM_BOT_SAFE_CRM_CONTEXT`:
  - в `direct_path.py` и `post_layers.py`;
  - не входит в `DIRECT_PATH_PILOT_PROFILE_DEFAULT_ON_FLAGS`;
  - runtime injection читает только env.
- `TELEGRAM_DIRECT_PATH_KNOWN_SLOTS_NEXT_STEP_PROMPT`:
  - входит в `DIRECT_PATH_PILOT_PROFILE_DEFAULT_ON_FLAGS`;
  - явный `0` перебивает профиль.
- `BOT_SAFE_MEMORY_STEP_GUARD_FLAG`:
  - это safety-метка `bot_safe_memory_unconfirmed_step_detected`, не env-переключатель.

## Паритет-smoke

Без вызова модели, детерминированно сравнивался direct prompt.

1. Голый `main`, оба env OFF:
   - сравнение `main 8ffb752` vs консолидация `185c4a0` (кодовая вершина до документационного отчёта);
   - 10/10 фиксированных входов побайтово совпали.
2. Под `pilot_gold_v1`, память OFF:
   - сравнение `codex/tz151-known-slots-next-step` vs консолидация `185c4a0` (кодовая вершина до документационного отчёта);
   - 10/10 фиксированных входов побайтово совпали.
3. Локальный ON sanity:
   - `TELEGRAM_BOT_SAFE_CRM_CONTEXT=1`;
   - prompt содержит `Безопасная выжимка клиента`;
   - prompt содержит `Следующий шаг: отправить расписание`.

Ограничение проверки: это prompt-level smoke без LLM-вызова, чтобы не зависеть от модели и не писать в live.

## NEG

- Память OFF не попадает в prompt:
  - `test_pilot_gold_keeps_memory_off_without_extra_context_flag`;
  - `test_bot_safe_context_prompt_is_default_off_even_when_context_present`.
- Безусловный provider-вызов гарда при памяти OFF — no-op:
  - `test_direct_path_bot_safe_memory_step_guard_is_noop_when_memory_off`.
- Helper-уровень гарда при OFF возвращает тот же объект:
  - `test_bot_safe_memory_step_guard_off_is_noop_with_memory_context`.
- Память ON локально доходит до prompt:
  - prompt-smoke выше.
- `next_step.status=needs_manager_review/empty` downgrade:
  - `test_bot_safe_memory_step_guard.py`.
- Над-квалификация:
  - `test_direct_path_known_slots_next_step_prompt.py`.

## Live safety

Не запускались:

- live AMO/Tallanto/CRM write;
- `--execute-live-write`;
- production apply полной памяти;
- nightly на prod DB;
- ASR/Resolve+Analyze;
- M1/измерительные машины.

Не тронуты:

- `stable_runtime`;
- `product_data/customer_timeline/customer_timeline_prod_*`;
- AMO/Tallanto;
- исходные YAML.

Важно для будущего включения: nightly CLI пишет в `timeline_db`, `journal_path` и опционально `profiles_db` только при явном запуске. На prod запускать только с отдельным «да» и бэкапом.

## Статус

Формально готово к регрейду Claude #1.
В `main` не переносил и не пушил.
