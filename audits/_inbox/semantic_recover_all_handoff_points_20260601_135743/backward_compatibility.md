# Backward compatibility

## Проверки

- `tests/test_dialogue_contract_pipeline.py tests/test_subscription_llm_draft_provider.py`: 434 passed.
- `tests/test_telegram_dynamic_client_sim.py tests/test_dialogue_memory.py`: 63 passed.
- Full `tests/`: 2381 passed, 9 failed due unrelated missing local artifacts / existing context expectations.

## Совместимость поведения

- Existing no-draft/draft-error fact-composer paths сохраняют автономный ответ, но теперь маркируются единым `fallback_reason="cite_only_recover"`.
- P0/refund/complaint paths не переводятся в автономный recover.
- Understanding prompt, P0 latch, route flip, semantic critic prompt and KB were not changed.
