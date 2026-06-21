# Semantic Review

Verdict: PASS_WITH_NOTES

Audience: Telegram bot draft context.

What passed:

- Bot receives only bot-safe summaries, not raw timeline events or customer profile.
- Next step is extracted from explicit summary/structured fields only; no LLM and no invented step.
- Names in `interest/title` are masked while known program and organization names are preserved.
- Brand visibility follows the approved rule: active brand plus unknown are visible, explicit foreign brand is excluded.
- Runtime PII scanner found no contact PII in visible bot-safe summaries.

Non-blocking risks:

- `interest/title` may still include high-risk business topics such as refund/bank references. This is not a PII leak, but the output P0 gate remains mandatory.
- `needs_manager_review` from contradictory later events is conservative and may reduce usefulness until separately analyzed.

Regression checks added:

- Names in title are masked.
- Program/org names are preserved.
- Person-only interest is dropped.
- Stale unknown chunk is retired after brand resolution.

Recommended next action:

- Claude #1 regread of audit pack, then D7 measurement on draft quality with bot-safe context enabled.
