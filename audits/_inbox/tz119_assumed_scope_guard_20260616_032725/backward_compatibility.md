# Backward compatibility

## Default behavior

`TELEGRAM_ASSUMED_SCOPE_GUARD` defaults to OFF.

When OFF:

- slot scope is built through the previous `_direct_path_slot_scope` behavior;
- prompts do not include slot status;
- model-driven retriever remains disabled unless the new guard is also ON;
- final draft text is not changed by `apply_assumed_scope_guard`.

## Intentional contract change

`TELEGRAM_RETRIEVER_MODEL_DRIVEN=1` alone no longer enables model-driven retrieval. This is intentional per TZ-119: model-driven retrieval must only run together with the anti-dodumka guard.

## Export compatibility

New names exported through `subscription_llm_parts.__init__`:

- `ASSUMED_SCOPE_GUARD_ENV`
- `apply_assumed_scope_guard`
- `_assumed_scope_guard_enabled`
- `_direct_path_slot_provenance`
- `_direct_path_soft_slot_scope`
