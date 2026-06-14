# Backward compatibility

OFF mode:

- `PROFILE_LLM_CHILD_RESOLVER` default remains OFF.
- Existing OFF NEG test remains green: old deterministic `apply_child_slot_merge_candidates` output matches.

ON mode:

- Prompt version changed from `v1` to `v3`; old LLM cache entries are intentionally bypassed.
- `merge_confidence` is required in model responses for v3 and is validated as `high|low`.
- `merge_confidence` does not affect merge/reject behavior in this iteration.
- Name-veto logic remains active: `different_child_names_merged` and `mention_name_misattributed` still fail-soft the family unchanged.

Production note:

- Because prompt changes can alter `canonical_name` and therefore `child_key`, any future production enablement must be a one-time full rebuild, not an incremental profile update.
