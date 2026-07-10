> DONE 2026-07-03 19:19 | ветка codex/adr003-semanticframe-migration | codex

> TAKE 2026-07-03 18:02 | ветка codex/adr003-semanticframe-migration | codex

Ветка: codex/adr003-semanticframe-migration
Зоны: src/mango_mvp/channels/subscription_llm_parts/semantic_reading.py, src/mango_mvp/channels/subscription_llm_parts/reliable_answerer.py, src/mango_mvp/channels/subscription_llm_parts/policy_routing.py, src/mango_mvp/channels/subscription_llm_parts/provider.py, src/mango_mvp/channels/dialogue_memory.py, scripts/run_telegram_dynamic_client_sim.py, scripts/build_adr003_env_matrix.py, scripts/run_adr003_semantic_reading_e3_paired.sh, tests/test_semantic_reading.py, tests/test_adr003_semantic_reading_trace.py, tests/test_adr003_semantic_reading_e3_runner.py, tests/test_adr003_regex_understanding_moratorium.py, tests/test_subscription_llm_draft_provider.py, tests/test_report_adr003_semantic_frame_eval.py, docs/ADR003_ETAP_T_DECISIONS.md, docs/ADR003_E3_ENV_MATRIX.md, docs/ADR003_DELETION_MANIFEST.md, tasks/_running/2026-07-03_TZ_ETAP_T_trace_i_reading_vrezki_repo_wrapper.md, audits/_inbox/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest --collect-only -q
Семантический-аудит: да

# Repo-wrapper: ADR003 этап T trace + reading-врезки

Источник ТЗ: `/Users/dmitrijfabarisov/Claude Projects/Foton/2026-07-03_TZ_ETAP_T_trace_i_reading_vrezki_dlya_D1.md`.

Цель: реализовать только этап T:
- `semantic_reading_trace` на direct-path без изменения поведения при OFF;
- три reading-врезки: `sense_seats`, `off_topic`, `slots_gsf`;
- hidden-хранилище `semantic_reading_slots`, без читателей и без утечки в prompt/known_slots;
- env-matrix и отдельный `e3_paired` runner для будущего замера;
- deletion manifest только как данные, без удаления regex.

Локальные правки к исходному ТЗ, принятые Codex перед реализацией:
- `docs/PROJECT_NOW.md` является generated/ignored файлом, поэтому обновляется локально через `scripts/project_now.py`, но не коммитится.
- Exact HEAD и обоснования решений фиксируются в tracked `docs/ADR003_ETAP_T_DECISIONS.md` и audit pack.
- Точка финализации trace должна быть после `apply_semantic_frame_decision_shadow`; старый `frame_decision_shadow` не переименовывать.
- Для `semantic_reading_slots` нужно явно определить способ чтения маски `slots_gsf`, не ломая существующий параметр `semantic_reading`.

Не делать:
- не трогать live-бота, AMO, Tallanto, CRM, Wappi;
- не менять профильный кортеж `DIRECT_PATH_PILOT_PROFILE_DEFAULT_ON_FLAGS`;
- не менять `scripts/run_adr003_semantic_reading_e2_triple.sh`;
- не объявлять env-заглушки Fix1b/Fix2/slots_reask;
- не включать маски в профиль;
- не удалять legacy regex.
