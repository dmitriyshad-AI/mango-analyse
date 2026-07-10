# Semantic Review

Status: semantic pass for report-only diagnostics.

The report answers a narrow business question: is there a broad class of safe handoff turns where the bot could give a partial factual answer now?

On the real 36ea110 enriched run:

- 44 handoff turns have some partial KB support.
- 0 are clean draft-route partial-answer candidates.
- 37 are excluded because the action/context is manager/payment/enroll/refund/handoff-like.
- 4 contain live availability or similar hard missing axes.
- 2 are `manager_only`, which is not lowerable without a separate owner policy.
- 1 has broad missing axes: proof exists for part of the answer, but dates/location/boarding remain uncovered.

Business conclusion:

- Do not enable F3 from partial-answer evidence.
- Do not lower `manager_only`.
- Do not generate client text from partial facts.
- The next useful step is a counterfactual action/proof calibration report, not active behavior.

PII/client text:

- No full candidate answer is generated or exported.
- Supporting facts are represented by keys/hashes/lengths in inherited helpers, not raw `client_safe_text`.
