Ветка: codex/adr003-semanticframe-migration
Зоны: src/mango_mvp/channels/dialogue_memory.py, scripts/run_telegram_dynamic_client_sim.py, tests/test_semantic_reading.py, tests/test_dynamic_client_sim_semantic_reading_memory.py, docs/ADR003_ETAP_T_DECISIONS.md, audits/_inbox/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest --collect-only -q
Семантический-аудит: да

# Wrapper PR-B slots_reask

Исполнять только PR-B из внешнего ТЗ:
`/Users/dmitrijfabarisov/Claude Projects/Foton/2026-07-03_TZ_FIX_zahod_fix1b_slots_reask_dlya_D1.md`.

Границы:

- сначала инвентаризация уже сделанного в текущем HEAD, не повторная реализация;
- `TELEGRAM_SLOTS_REASK` сам не создаёт hidden slots, а только читает уже записанные `semantic_reading_slots`;
- hidden slots создаются только при активном `TELEGRAM_SEMANTIC_READING_CLASSES=slots_gsf`;
- в `do_not_ask_again` попадают только имена слотов, не значения;
- `to_prompt_view()` не должен содержать `semantic_reading_slots` или SENTINEL-значения;
- проверить три sim/update точки `update_dialogue_memory_after_answer(... semantic_reading=...)`;
- PR-A/Fix1b, PR-C, known_slots merge и legacy G/S/F regex не трогать.
