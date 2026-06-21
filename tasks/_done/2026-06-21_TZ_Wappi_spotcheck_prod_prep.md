> DONE 2026-06-21 17:03 | ветка codex/wappi-history | codex

> TAKE 2026-06-21 16:54 | ветка codex/wappi-history | codex

Ветка: codex/wappi-history
Зоны: tasks/, audits/, product_data/customer_timeline/, docs/worktrees_registry.md
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_wappi_history_import_to_timeline.py tests/test_amo_wappi_auto_resolver.py tests/test_amo_wappi_transport.py tests/test_run_amo_wappi_draft_loop.py
Семантический-аудит: да

# ТЗ: Wappi spot-check и подготовка боевого долива

## Цель

Завершить Wappi-трек read-only проверкой перед боевым доливом:

1. Подтвердить две привязанные Wappi-сессии: реально один клиент, бренд канала совпадает с брендом сделки, привязка трассируема.
2. Зафиксировать входной курсор/снимок тестовой заливки без ПДн.
3. Подготовить чек-лист боевого долива с бэкапом, проверками и откатом.
4. Сам боевой долив не запускать.

## Границы

- Wappi только read-only, отправок нет.
- AMO только read-only через MCP/GET-only клиент.
- Tallanto/CRM/AMO write запрещены.
- Боевую timeline-БД не менять.
- Токены, полные телефоны, email, raw chat ids и raw message ids не писать в git/отчёт.
- `product_data/customer_timeline/` используется только для ignored cursor-артефакта текущей тестовой копии.

## Приёмка

- В отчёте есть 2 обезличенные трассы привязанных сессий.
- Подтверждены: один customer_id в сессии, один lead/contact, бренд сделки совпадает, `allowed_for_bot=0`.
- Создан ignored cursor manifest с sha256 и описан способ сравнения перед apply.
- Есть чек-лист production apply: preflight, backup, dry-run, apply по отдельной отмашке, readback, rollback.
- NEG: production apply не запускался; боевую БД не трогали; отправок Wappi нет; AMO/Tallanto write нет; ПДн в отчёте нет.
