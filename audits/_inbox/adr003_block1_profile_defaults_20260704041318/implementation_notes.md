# Что сделано

- Блок 1 мега-ТЗ: профиль `pilot_gold_v1` теперь включает inline `TELEGRAM_SEMANTIC_FRAME_SHADOW` через профильный default-on механизм.
- Для `TELEGRAM_SEMANTIC_READING_CLASSES` добавлен отдельный CSV-дефолт профиля: `sense_seats,slots_gsf,off_topic`.
- Явный override сохранен: context/env значение имеет приоритет; пустая строка выключает классы.
- Добавлен repo-local wrapper ТЗ для ночного марафона и взят через `task_move.py`; `preflight.py` прошел.

# Как проверялось

- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_direct_path_semantic_frame_shadow.py tests/test_semantic_reading.py tests/test_adr003_regex_understanding_moratorium.py` -> `71 passed`.
- Runtime resolution:
  - context profile -> `frame_shadow=True`, classes `off_topic/sense_seats/slots_gsf`;
  - explicit empty classes -> `[]`;
  - process-env profile with no classes env -> `reading_class_enabled(None, "slots_gsf") == True`;
  - process-env empty classes -> `False`.
- Local profile smoke: `runs/adr003_profile_smoke_76f80736_20260704_040503`.
  - Dynamic summary: `20 dialogs / 20 turns`, `ok=true`, `hard_gate_failures=0`.
  - Validation with `--expect-trace`: `frames=20`, `trace_turns=20`, `eligible_frame_rate=1.0000`, `timeouts=0`.
- Full pytest: `4018 passed, 5 skipped, 1 warning`.

# Что осталось

- Приемочный semantic_pass не выносился: smoke направленческий, утренний регрейд по сырью остается за Fable/Claude.
- Следующий блок: PR-A Fix1b через отдельный repo-local wrapper.
