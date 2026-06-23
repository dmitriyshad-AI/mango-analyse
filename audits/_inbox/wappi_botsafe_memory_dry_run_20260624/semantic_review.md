# Semantic Review

Status: `formal_pass`, `semantic_risk`.

What is safe:

- Bot-safe memory is taken only from structured bot-safe chunks.
- `unknown` brand chunks are not passed into a brand-specific Wappi prompt.
- Foreign-brand chunks are not passed.
- Ambiguous identity returns empty context.
- PII-like phone/email/service ids in summary text are blocked.
- Trash-like placeholder summaries are not passed.
- Wappi journal metadata does not contain raw customer memory.

Residual semantic risk:

- The actual Wappi dry-run had only one recent paired chat with bot-safe memory, so behavior must be reviewed on a wider dry-run sample.
- The draft generator can still ask broad clarifying questions when memory exists but lacks exact slots such as class, subject, or format.
- Strict `unknown` filtering intentionally reduces coverage until useful unknown memory is re-tagged by brand.

No production-ready verdict is made here.
