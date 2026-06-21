# D3 Phase 0.1/0.2 Bot-Safe Integration Report

Дата: 2026-06-21
Ветка: `codex/d3-phase01-botsafe-integration`
База проверки: свежая тест-копия боевой памяти `/tmp/mango_phase01_integration_final_20260621_102959/customer_timeline.sqlite`

## Что интегрировано

- Влит D8-коммит `7387998` как локальный cherry-pick `e268f40`: детерминированный извлекатель следующего шага из summary звонков.
- Сохранена бренд-видимость D3: бот видит `active_brand + unknown`, явный чужой бренд не подмешивается.
- Добавлена очистка имён в `interest/title` bot-safe выжимки:
  - маскируются роль+имя, ФИО/имя+фамилия, одиночные частые имена;
  - сохраняются названия программ/организаций/брендов: ЛВШ, летняя/зимняя/очная школа, Интенсив Мат/Физ/Инф/Рус, Альфа Банк, Фотон, УНПК, МФТИ, ЕГЭ, ОГЭ, М9, М11.
- Добавлено выключение устаревших bot-safe chunks: если `botsafe:{customer}:unknown` больше не является актуальной выжимкой после уточнения бренда, запись не удаляется, но становится `allowed_for_bot=0`, `requires_manager_review=1`.

## Финальные метрики на свежей тест-копии

До финального прогона:

- видимых bot-safe chunks: `17856`;
- `next_step`: `empty=17856`;
- известный бренд по сделкам/событиям: `foton=1290`, `unpk=4017`, `unknown=12549`;
- Telegram УНПК: `1377` событий, `202` клиента.

После финального прогона:

- видимых bot-safe chunks: `17901`;
- бренды: `foton=1290`, `unpk=4162`, `unknown=12449`;
- источники бренда: `deal=1348`, `event=4104`, `unknown=12449`;
- следующий шаг: `active=2582`, `closed=2011`, `empty=10502`, `needs_manager_review=2806`;
- устаревшие видимые chunks выключены: `retired_stale=100`;
- Telegram УНПК клиенты с видимой выжимкой: `202/202`;
- `raw_allowed_chunks_after=0`;
- runtime PII findings: `{}`;
- имена в `interest/title` после исключения названий программ: `{}`;
- чужой бренд в видимом тексте: `{}`.

Идемпотентность:

- повторный прогон: `created=0`, `updated=0`, `retired_stale=0`, `duplicate=17901`.

## NEG-проверки

- Боевая БД не менялась: `sha256`, размер и `mtime` до/после совпали.
- Запись выполнялась только в тестовую копию в `/tmp`.
- Контактные ПДн в видимой bot-safe памяти: `0` по runtime scanner.
- Реальные имена в `interest/title`: `0` по скану после исключения безопасных названий.
- Названия сохранены: летняя/зимняя выездная школа, ЛВШ, Альфа Банк.
- Явный чужой бренд скрыт: проверено на 5 клиентах с двумя брендами через runtime helper.
- Unknown-чанк виден активному бренду: проверено на 3 клиентах unknown-only.
- Исходные raw chunks остаются неразрешёнными для бота: `raw_allowed_chunks_after=0`.

## Тесты

- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_customer_timeline_bot_safe_summary.py tests/test_customer_timeline_next_step_resolver.py tests/test_bot_safe_runtime_context.py tests/test_bot_safe_direct_path_context.py`
  - `43 passed`
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q`
  - `3513 passed, 5 skipped, 1 warning`

## Остаточные риски

- В `interest/title` всё ещё могут быть финансово-юридические темы вроде возврата или банка. Это не ПДн и не бренд-утечка; задача текущего среза была про имена и бренд-видимость. P0/бренд-гейт на выходе должен оставаться обязательным.
- `needs_manager_review=2806` в следующем шаге вызван `contradictory_later_event`; это безопасный режим, но требует отдельного анализа качества, если таких кейсов окажется слишком много для черновиков.

## Артефакты

Audit pack: `audits/_inbox/d3_phase01_botsafe_integration_20260621_1333/`
