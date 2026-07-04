# PR-B slots_reask inventory

Блок 3 не переизобретался: PR-B уже реализован в текущем HEAD через исторический коммит `10bb2f6e feat(adr003): add semantic slots reask shadow`. В этом блоке сделана инвентаризация, добавлен repo-local wrapper и зафиксировано решение D-014.

## Инвентаризация

| Пункт ТЗ | Статус | Сырьё |
|---|---|---|
| Env `TELEGRAM_SLOTS_REASK` default-OFF | есть | `src/mango_mvp/channels/dialogue_memory.py:34`, helper `_slots_reask_enabled()` `:1809` |
| Hidden slots создаются только при `slots_gsf` | есть | `_semantic_reading_slots_from_payload()` проверяет `reading_class_enabled(None, "slots_gsf")` `dialogue_memory.py:1569` |
| `SLOTS_REASK` сам hidden slots не создаёт | есть | тест `test_slots_reask_does_not_create_hidden_slots_without_slots_gsf` |
| `SLOTS_REASK` + semantic payload при выключенном `slots_gsf` остаётся no-op | есть | тест `test_slots_reask_with_semantic_payload_is_noop_when_slots_gsf_is_off` |
| N-1 merge имён hidden slots в `do_not_reask` | есть | `build_dialogue_memory()` `dialogue_memory.py:444-448` |
| Values не попадают в prompt | есть | `DialogueMemory.to_prompt_view()` не отдаёт `semantic_reading_slots`; тесты sentinel-leak |
| Пустые values не попадают в `do_not_reask` | есть | `_semantic_reading_slot_names()` + `test_slots_reask_ignores_empty_hidden_values_and_never_leaks_sentinels` |
| Memory LLM update не раскрывает hidden values | есть | `_apply_memory_llm_update()` `dialogue_memory.py:657-661`; тест `test_slots_reask_survives_memory_llm_update_without_value_leak` |
| Sim/update point 1 пробрасывает reading | есть | `attach_context_facts_to_dialog()` `scripts/run_telegram_dynamic_client_sim.py:1856-1863` |
| Sim/update point 2 пробрасывает reading | есть | `enrich_transcripts_with_semantic_frame()` `scripts/run_telegram_dynamic_client_sim.py:1992-1999` |
| Sim/update point 3 пробрасывает reading | есть | `run_one_dialog()` `scripts/run_telegram_dynamic_client_sim.py:2087-2094` |
| Static guard на 3 sim-точки | есть | `tests/test_dynamic_client_sim_semantic_reading_memory.py` |

## Решение блока

Runtime-код не менялся. Добавлены:

- repo-local wrapper `tasks/tz_pr_b_slots_reask_wrapper.md`;
- D-014 в `docs/ADR003_ETAP_T_DECISIONS.md`, который уточняет: PR-B является hidden-storage/read-only anti-reask механизмом, не merge в `known_slots`.

## Оговорка по live-пути

Live `draft_loop` сохраняет post-answer memory только при включённом memory-provenance режиме. Это не меняет безопасность PR-B, но для будущего включения `TELEGRAM_SLOTS_REASK` нужно проверять матрицу: `MEMORY_PROVENANCE`/profile ON + `slots_gsf` пишет hidden slots, затем `SLOTS_REASK` читает только имена.
