> DONE 2026-07-03 03:33 | ветка codex/adr003-semanticframe-migration | codex

> TAKE 2026-07-03 02:21 | ветка codex/adr003-semanticframe-migration | codex

Ветка: codex/adr003-semanticframe-migration
Зоны: src/mango_mvp/channels/subscription_llm_parts/, src/mango_mvp/channels/dialogue_memory.py, tests/, scripts/, docs/, tasks/, audits/_inbox/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_adr003_regex_understanding_moratorium.py tests/test_direct_path_semantic_frame_shadow.py tests/test_subscription_llm_draft_provider.py tests/test_dialogue_memory.py tests/test_report_adr003_semantic_frame_eval.py tests/test_semantic_reading.py
Семантический-аудит: да

# Рабочая обёртка ТЗ: semantic_reading один блок

Источник ТЗ: `/Users/dmitrijfabarisov/Claude Projects/Foton/2026-07-03_TZ_USKORENIE_semantic_reading_odin_blok_dlya_D1.md`.

Причина обёртки: оригинальный файл Foton остаётся общим источником для Дмитрия и Claude #1. Не перемещать и не удалять его через `task_move.py`.

Исполняемый объём в этом рабочем дереве:

- Э0: Ф0a + Ф0b из исходного ТЗ: удалить 9 мёртвых regex и расширить moratorium guard на regex/keyword/marker-helper понимание, включая тяжёлых marker-helper потребителей из обновлённого Foton-ТЗ.
- Э1: `semantic_reading.py`, inline/posthoc source metadata, default-OFF, `off_topic` в трёх точках, контракт LLM-слотов по словарю П1c: только closed dictionary + client-authored history floor + `source_name="semantic_reading_llm"`.
- Э2: подготовить расширение отчёта/среза для M1, если Э1 закрыт тестами.

Не исполнять без отдельного регрейда и нового решения Дмитрия:

- Э3-A: включение масок классов.
- Э3-B: удаление legacy-regex зелёных классов.
- Любое изменение live/profile/P0-floor/P0-preblock/tone-close/live-status floor.
