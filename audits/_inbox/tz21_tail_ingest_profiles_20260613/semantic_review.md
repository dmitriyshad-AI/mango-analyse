# TZ-21 semantic review

Verdict: `PASS_WITH_NOTES`

## Artifact and audience

Artifact: deterministic customer profile rebuild after importing the 3,439-call analyze tail.

Audience: internal analyst / CRM profile consumer. Not customer-facing text.

## What passed

- The rebuild uses the same customer timeline source and updated canonical call summaries.
- v7 coverage in the target zone increased exactly by 3,439 calls.
- The remaining long old calls in zone are the 56 blacklist calls, which TZ-21 explicitly must not import.
- The 5 examples are anonymized: hashes, counts, field names, brand/source sets, lengths only.
- No names, phones, raw quotes, transcripts, or CRM-ready text were written into tracked reports.
- AMO/Tallanto/CRM write operations were not performed.

## Non-blocking risks

- This stage does not adjudicate `needs_review` calls. Those flags remain as imported.
- Active field count decreased by 121 even though total fields increased by 56. This is plausible after replacing old summaries with stricter v7 summaries and superseded-field rules, but it should be watched in downstream CRM-card review.
- Some high-event profiles still have zero active fields. That is a data coverage issue, not a rebuild failure.
- `build_tz16_profiles_v7.py` still has historical `tz16` names in function/schema/build ids. The TZ-21 output root and report make the actual release clear, but a later cleanup should rename the reusable builder.

## Missing checks

- No manual business review of real profile contents was performed because tracked reports must not expose personal data.
- No CRM card stage B was run; it remains outside TZ-21.

## Required follow-up gates

- Keep anonymized-only examples in tracked reports.
- Before CRM-card use, review a small private sample from the ignored profile DB.
- Treat the 56 blacklist calls as TZ-20/TZ-blacklist scope, not as a TZ-21 failure.
