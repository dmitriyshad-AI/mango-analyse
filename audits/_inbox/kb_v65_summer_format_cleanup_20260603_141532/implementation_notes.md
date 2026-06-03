# KB v6.5 summer format cleanup

Дата: 2026-06-03

Сделано:
- взят актуальный source-пакет `kb_release_20260602_v6_4_schedule_sources` как база, чтобы v6.5 наследовал schedule-факты;
- создан source-пакет `product_data/knowledge_base/kb_release_20260603_v6_5_summer_format_cleanup_sources`;
- добавлено 11 client-safe фактов из `KB_DRAFT_lesson_format_and_summer_schools_2026-06-03.md` дословно: 4 для Фотона, 7 для УНПК;
- синхронизированы старые структурные ЛШ-значения, чтобы не оставить противоречия: УНПК base 39 500 вместо 37 500, Пацаева как предзапись, действующие/бывшие ученики 5% вместо старых 7%;
- очищены 24 `objection_responses` client-safe факта от префикса `черновик для ситуации ...`, ключи `objection_responses.*` сохранены;
- сборка выполнена только через `scripts/build_kb_release_v6_1_team_answers.py`;
- `DEFAULT_KB_SNAPSHOT_PATH` не менялся.

Новый релиз:
- `product_data/knowledge_base/kb_release_20260603_v6_5_summer_format_cleanup`
- `product_data/knowledge_base/kb_release_20260603_v6_5_summer_format_cleanup_bot_pack`
- `product_data/knowledge_base/kb_release_20260603_v6_5_summer_format_cleanup_employee_pack`
- `product_data/knowledge_base/kb_release_20260603_v6_5_summer_format_cleanup_handoff_for_claude_and_team`

