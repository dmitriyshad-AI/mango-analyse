> DONE 2026-06-21 01:28 | ветка codex/etap3-botsafe-layer | codex

> TAKE 2026-06-21 00:26 | ветка codex/etap3-botsafe-layer | codex

Ветка: codex/etap3-botsafe-layer
Зоны: src/mango_mvp/customer_timeline/, scripts/, tests/, tasks/, docs/worktrees_registry.md, audits/_inbox/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_customer_timeline_bot_safe_summary.py tests/test_customer_timeline_store.py tests/test_customer_timeline_context_provider.py
Семантический-аудит: да

# ТЗ — Этап 3, Фаза 0: bot-safe слой

Источник: `/Users/dmitrijfabarisov/Claude Projects/Foton/2026-06-21_TZ_Etap3_Faza0_bot_safe_sloy.md`, v2.

## Ключевые требования

- Работать от ревизии `716cbcd`.
- Работать только на тестовой памяти моста; в боевую timeline/AMO/Tallanto/CRM/YAML/stable_runtime не писать.
- Создать один безопасный `bot_context_chunk` на клиента: `chunk_type=bot_safe_summary`, `source_ref=botsafe:{customer_id}`, `allowed_for_bot=1`, `requires_manager_review=0`.
- Выжимка только из структурных полей:
  - бренд: `customer_opportunities.record_json.product_context.brand`;
  - стадия: `customer_opportunities.status`;
  - интерес: `product_context.products_of_interest` / `title`;
  - следующий шаг: D8 `resolve_customer_next_step` на лету.
- Сырой `bot_context_chunks.text` / `summary` исходных сводок не копировать.
- Любые фрагменты `title` пропускать через `redact_text`.
- Бренд резолвить на уровне клиента; при неизвестном бренде не добавлять бренд-специфику.
- NEG: email/телефон в bot-safe тексте = 0; сырой текст не скопирован; чужой бренд не включён; исходные чанки остаются `allowed_for_bot=0`; повтор не плодит дубли.
