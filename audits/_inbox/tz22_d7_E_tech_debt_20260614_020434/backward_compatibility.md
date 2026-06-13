Backward compatibility

Default behavior:
- PROFILE_PHONE_INDEX OFF keeps old profile schema and old full-scan lookup path.
- Deal-aware explicit analysis_date values keep the previous output date.
- Existing analysis_json readers keep analysis_meta and quality_flags.

NEG:
- Profile DB created with PROFILE_PHONE_INDEX unset has no primary_phone_norm column.
- resolve_analysis_date("2026-05-13") returns "2026-05-13".
- Phone wrappers retain prior output shapes: insight digit-only, telegram/mail/timeline plus-format or empty/None as before.

Changed behavior:
- Canonical phone normalizer no longer turns short junk into a phone key.
- New deal-aware paths with analysis_date=None use current UTC date.
- New Analyze writes include top-level prompt/model metadata.

Unaffected:
- Stage20 live token string is unchanged and marked historical.
- Dead-code candidates are documented only; no script deletion.
