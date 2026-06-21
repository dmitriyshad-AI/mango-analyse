> DONE 2026-06-21 15:18 | ветка codex/wappi-history | codex

> TAKE 2026-06-21 14:18 | ветка codex/wappi-history | codex

Ветка: codex/wappi-history
Зоны: src/mango_mvp/customer_timeline/, src/mango_mvp/integrations/, scripts/import_wappi_history_to_timeline.py, scripts/run_amo_wappi_draft_loop.py, tests/test_wappi_history_import_to_timeline.py, tests/test_run_amo_wappi_draft_loop.py, tests/test_amo_wappi_transport.py, tests/test_amo_wappi_auto_resolver.py, tasks/, audits/, docs/worktrees_registry.md
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_wappi_history_import_to_timeline.py tests/test_amo_wappi_auto_resolver.py tests/test_amo_wappi_transport.py
Семантический-аудит: да

# ТЗ — Wappi: привязка истории к клиентам через авто-резолвер

Источник: `/Users/dmitrijfabarisov/Claude Projects/Foton/2026-06-21_TZ_Wappi_privyazka_rezolver.md`.

Задача: на тестовой копии памяти привязать Wappi-историю к клиентам памяти через тот же read-only AMO-резолвер, что используется в живом draft-loop контуре, без записи во внешние системы и без записи в боевую память.

Границы:
- Wappi только read-only через `DefaultDenyTransport`.
- AMO только read-only через GET-only `AmoMcpClient`.
- Запись в AMO/Tallanto/CRM запрещена.
- Боевую память, `stable_runtime`, YAML и live-бот не трогать.
- `auto_pairs_path` на тест-прогоне не писать.
- Wappi-события всегда `allowed_for_bot=0`.

Работа:
1. Диагностировать покрытие pending Wappi-чата: TG `chat_id.isdigit()`, точный матч Telegram ID, `username_only`; MAX телефон/стоп-лист.
2. Перенести `AmoAutoResolver` из `scripts/run_amo_wappi_draft_loop.py` в общий модуль и использовать его в Wappi-импортёре для чатов без статичной пары.
3. Цепочка привязки: chat -> AMO lead/contact -> существующий мост памяти -> `customer_id`; неоднозначность -> pending с причиной.
4. Повторить тестовую заливку в тест-копию, проверить разбивку причин и идемпотентность.
5. Добавить NEG-тесты: TG не по телефону, MAX только со стоп-листом, бренд сделки совпадает с брендом канала, повтор не переклеивает клиента молча, записи/отправки невозможны.

Отчёт: в `tasks/_done/` с диагностикой покрытия, счётчиками привязки/pending по причинам, 3-5 обезличенными трассировками, NEG, тестами, веткой и коммитом.
