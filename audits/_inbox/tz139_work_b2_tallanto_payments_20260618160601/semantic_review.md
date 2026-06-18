# Semantic Review

Status: semantic_self_review_done, pending Claude independent regrade.

## Checks

- Payment and abonement facts are kept as internal timeline facts, not bot-safe customer-facing context.
- Exact amounts and remaining visits are stored only in `TimelineEvent.record` and `CustomerOpportunity.product_context`.
- No `BotContextChunk` is created for Tallanto payment/abonement rows, so bot-safe context cannot expose sums from B2.
- Ambiguous Tallanto contact ids do not merge into the first candidate; they create a conflict.
- Unmatched Tallanto contact ids become partial/unmatched local identities, not strong CRM customer facts.
- Free text raw fields such as `description`, `contact_notice`, `internal_notice`, and raw MCP payload wrappers are excluded from the stored projection.

## Remaining Semantic Risks

- The real-data sample is limited to 50 rows per module because the read-only MCP tool enforces `limit <= 50`; Claude should regrade on broader source if needed.
- The importer does not call Tallanto directly. Operational workflow must keep `crm_call.sh` output outside git/audit unless scrubbed.
- Existing generic store behavior reports `updated` on repeated apply for some upsert/mapping records. Event row counts remain stable, but Claude should judge whether this is acceptable for B2 idempotency.

## Verdict

Formal tests and self semantic review pass. This is not final semantic_pass until Claude regrades.
