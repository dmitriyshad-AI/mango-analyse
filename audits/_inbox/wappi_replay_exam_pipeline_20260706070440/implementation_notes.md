# Что сделано

- Добавлен пакет `mango_mvp.replay_exam`:
  - read-only Wappi exporter с пагинацией и `mark_all=False`;
  - recursive pseudonymizer;
  - teacher-forcing slicer с `chat_only/external_context/manager_issue_private`;
  - offline runner с параллельностью только по диалогам;
  - machine gate для чисел, брендов, ПДн и P0 route/flags;
  - `replay_judge_v1` payload с hidden A/B key;
  - M1 manifest adapter.
- Добавлены CLI:
  - `scripts/export_wappi_replay_dialogs.py`;
  - `scripts/run_wappi_replay_exam.py`;
  - `scripts/build_wappi_replay_m1_manifest.py`.
- Добавлен документ `docs/WAPPI_REPLAY_EXAM_PIPELINE.md`.
- Решение D-044 записано в `docs/ADR003_ETAP_T_DECISIONS.md`.

# Как проверялось

- `pytest tests/test_wappi_replay_*.py`: 15 passed
- `py_compile replay modules/scripts`: OK
- offline fake replay smoke: 3 turns, machine_gate_failures=0, `llm_calls.client=0`
- M1 manifest smoke: metric=`chat_only_replay`
- `pytest --collect-only`: 4165 tests collected
- `git diff --check`: OK

# Что осталось

- Реальный Wappi export и pilot-10 не запускались: это live-read/ПДн зона и требует отдельного явного подтверждения владельца.
- Реальный provider adapter для `run_wappi_replay_exam.py` пока не подключён: CLI разрешает только `--fake-provider`, чтобы не сделать скрытый live/LLM прогон без методического GO.
