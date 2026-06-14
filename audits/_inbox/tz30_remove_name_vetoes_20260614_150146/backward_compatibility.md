# Backward compatibility

OFF mode:

- `PROFILE_LLM_CHILD_RESOLVER` default remains OFF.
- Existing OFF NEG remains green: old deterministic `apply_child_slot_merge_candidates` output matches.

ON mode:

- Prompt version changed to `v4`; previous LLM cache is intentionally bypassed.
- `merge_confidence` no longer rejects a family when missing/invalid; local normalize defaults it to `low`.
- `child_key` is stable by mention group, not by model-selected canonical name.
- Name identity is no longer checked by deterministic rules; model decision plus mechanical guards define ON behavior.

Production note:

- Future production enablement remains one-time full rebuild only, not incremental, because resolver behavior changes child grouping.
