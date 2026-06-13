Risk review

Primary risks addressed:
- Divergent phone normalization causing missed customer/profile joins.
- Full-table profile phone scan remaining the only possible path.
- New deal-aware runs silently using stale May 2026 dates.
- Analyze outputs lacking direct prompt/model/truncation metadata.
- Suspected dead code being deleted without owner decision.

Guardrails:
- PROFILE_PHONE_INDEX defaults OFF.
- Explicit date tests preserve old behavior.
- Phone wrapper tests preserve old output formats per module.
- Analyze metadata is additive only.
- E5 deletes nothing.

Known gaps:
- No source snapshot key-drift measurement was run for E1.
- No production profile DB ALTER/backfill was run for E2.
- No analyze job was executed for E4.
