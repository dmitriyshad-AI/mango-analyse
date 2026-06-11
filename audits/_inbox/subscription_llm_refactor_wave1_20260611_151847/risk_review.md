# Risk review

Remaining risks:

- Baseline now includes 19 deterministic cases, not the final 89-scenario acceptance replay. The 89-case parallel replay is reserved for after wave 8.
- `subscription_llm_parts` is absent in wave 1, so identity checks are skipped in monolith mode. They become strict when the package appears.
- Coverage parity is proxy-only in wave 1. Strict parts coverage starts with the first move wave.
- Replay intentionally uses fake runners and local subclasses on approved seam points; it does not prove live LLM behavior.
- Prompt timestamps are fixed only in prompt-builder/dialogue-memory dependencies for deterministic measurement; `subscription_llm` facade and future parts keep exploding sentinels for `datetime.now`/`date.today`.

Hard controls added:

- frozen export snapshot prevents disappearing names from being silently accepted;
- move-only AST/compiled-body snapshot fails any changed function/class body;
- replay fails unexpected provider/runtime/semantic fallback markers unless a case explicitly allows fallback/policy downgrade;
- default `DIRECT_PATH_REAL_MANAGER_GOLD_PACK_PATH` is asserted to exist without env override;
- `_guard_cache_dir` stable_runtime rejection is covered.
