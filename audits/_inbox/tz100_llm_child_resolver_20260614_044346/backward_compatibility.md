# Backward compatibility

Default behavior:
- `PROFILE_LLM_CHILD_RESOLVER` defaults to OFF.
- OFF path still uses the existing deterministic child-slot merge logic.
- Existing `PROFILE_CHILD_MERGE_BY_TRAIT` behavior is unchanged and remains default OFF.

Database/write compatibility:
- No schema change.
- Rekey changes only `child_key` for child fields when the resolver accepts a family.
- Source refs, event timestamps, values, quotes, and brand provenance are preserved.
- `field_id` is recalculated only when `child_key` changes, matching the existing rekey pattern.

Runtime boundaries:
- No AMO/CRM/Tallanto writes.
- No ASR.
- No Resolve+Analyze.
- Full 7,512-family run not executed.

Verification:
- Full pytest passed.
- Microprobe before/after SQLite quick_check passed.
- Ignored microprobe output confirmed through `.gitignore:141`.
