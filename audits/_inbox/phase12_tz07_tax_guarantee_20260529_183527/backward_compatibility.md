# Backward Compatibility

- Legacy `_tax_safe_template` не переписан; v2 producer только уточняет выбор безопасной ветки перед fallback в legacy-селектор.
- Существующий `test_tax_amount_question_uses_amount_formula_template` проходит.
- Block-A tax follow-up тесты проходят в прицельном наборе.
- Общий файл `tests/test_subscription_llm_draft_provider.py` проходит полностью.
- `tests/test_dialogue_contract_pipeline.py` проходит полностью.
- TZ-08/TZ-09, Phase 2/KB и refund-latch не затронуты.
