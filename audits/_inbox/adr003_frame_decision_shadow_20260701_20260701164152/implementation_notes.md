# Что сделано
- Создан default-OFF флаг `TELEGRAM_SEMANTIC_FRAME_DECISION_SHADOW`.
- В direct-path ветке после `apply_tone_close_detect_layer` и финального `scrub_direct_path_p0_text` добавлен no-op слой `apply_semantic_frame_decision_shadow`.
- Слой читает уже готовый `semantic_frame`, итоговый route, `action_decision`, `close_detect`, safety flags/direct metadata и пишет только `metadata.frame_decision_shadow` + копию в `metadata.direct_path.frame_decision_shadow`.
- Добавлена сводка `frame_decision_shadow` в `scripts/run_telegram_dynamic_client_sim.py`: counts observed/no_frame и match/mismatch по handoff/P0/answerability/close/action.
- Поведение `route`, `draft_text`, `safety_flags`, `manager_checklist` и число model calls не меняются.

# Как проверялось
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_direct_path_semantic_frame_shadow.py tests/test_direct_path_payment_refund_split.py tests/test_adr003_regex_understanding_moratorium.py tests/test_telegram_dynamic_client_sim.py::test_summary_includes_frame_decision_shadow_counts` -> 17 passed.
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_subscription_llm_draft_provider.py tests/test_direct_p0_text_hygiene.py tests/test_reliable_answerer_step1.py` -> 566 passed.
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q` -> 3768 passed, 5 skipped, 1 warning.
- `git diff --check` -> clean.
- Graphify stats: карта свежая на `4c90e76`, использовалась только как навигация; выводы проверены исходниками.
- Финальный read-only аудитор: PASS; блокеров нет. Его неблокирующее замечание принято: тест теперь явно сравнивает `safety_flags` и `manager_checklist` ON против OFF.

# Что осталось
- Wappi latest25 пока не является прямым `--scenarios` набором для `run_telegram_dynamic_client_sim.py`; нужен адаптер или отдельный replay harness.
- Для per-field accuracy нужны gold labels `expected_frame`; текущий слой считает alignment с действующими детекторами, а не истинную точность.
- Активное включение close/action/relevance по SemanticFrame не делалось и должно ждать качества frame на eval.
