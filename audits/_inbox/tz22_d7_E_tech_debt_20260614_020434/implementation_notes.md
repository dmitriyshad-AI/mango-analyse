TZ22 D7 block E1-E5: CRM layer technical debt

E1:
- Updated canonical mango_mvp.utils.phone.normalize_phone.
- Canon now rejects short junk, handles 10-15 digit phones, 8->7 Russian conversion, Excel/scientific notation, and explicit international + prefixes.
- Replaced four local implementations with wrappers around the canonical normalizer while preserving their output contracts:
  - insights: digit-only or None.
  - mail_archive: +format or empty string.
  - telegram_history: +format or None.
  - customer_timeline context provider: +format or empty string.

E2:
- Added PROFILE_PHONE_INDEX default OFF.
- When ON, CustomerProfileSQLiteStore adds primary_phone_norm and idx_customer_profiles_phone_norm.
- CRM summary phone lookup uses the indexed column only when the flag is ON and the column exists; otherwise it falls back to the old full scan.

E3:
- DealTextPaths, DealQualityGatePaths, and DealAwareStage6Paths now default analysis_date to None.
- Added resolve_analysis_date(): explicit dates stay unchanged; None resolves to current UTC date.
- Replaced the hardcoded 2026-05-13 follow-up fallback.
- Kept the historical stage20 token and marked it as historical.

E4:
- Analyze now writes additive top-level metadata for future runs:
  - analyze_model
  - analyze_prompt_profile
  - analyze_prompt_truncated
  - analyze_prompt_chars
- Existing quality_flags and analysis_meta remain unchanged.

E5:
- Added dead_code_candidates.md under audits/_inbox/crm_layer_audit_2026-06-13.
- No files were deleted.

Out of scope:
- No schema migration of production profile DB.
- No analyze rerun.
- No live AMO/Tallanto writes.
