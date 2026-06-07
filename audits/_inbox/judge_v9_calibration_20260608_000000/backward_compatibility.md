# Backward compatibility

- Default judge prompt remains `v2`.
- Existing runner invocation without `--judge-prompt-version` preserves the current measurement series.
- Existing `judge_version` remains tied to `JUDGE_FACT_AUDIT_VERSION`; the new prompt version is recorded separately.
- Existing `--transcripts-in` behavior is unchanged.
- The new offline sidecar script does not modify old transcript or judge files.
- `fact_claim_audit.audit_fact_claims` keeps runtime behavior by default.
- No bot draft, route, gate, KB, CRM, or live-write logic was changed.
