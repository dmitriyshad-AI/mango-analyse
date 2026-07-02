# Semantic Review

Status: semantic pass for report-only diagnostics.

The report supports Claude's correction: the current autonomy loss is not a clean `harmless ack/status` route-only class. On the real 36ea110 artifacts the current handoff rows split as:

- 2 danger-adjacent rows: keep excluded from autonomy.
- 1 proof-axis mismatch: available proof does not cover all requested missing-fact axes.
- 1 frame calibration problem: a safe age/existence question is classified as `check_availability`.
- 1 partial-fact case: platform and price facts are present, but grade is still missing.

Business conclusion:

- Do not activate F3 from this evidence.
- Do not lower `manager_only`.
- Do not build client text from partial facts until the owner approves a separate partial-answer policy.
- Next work should be a shadow-only calibration/fact-proof step.

PII/client text:

- The report stores redacted excerpts only.
- Supporting facts are represented by keys, hashes, valid-until and lengths, not raw `client_safe_text`.
