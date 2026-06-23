# Implementation Notes

Task: connect existing bot-safe Customer Timeline memory to the Wappi -> AMO draft loop in read-only/dry-run mode.

Implemented:

- strict active-brand matching in `bot_safe_runtime_context`;
- no `unknown` fallback for active brand;
- no `unknown` relevance tags in the prompt-facing items;
- filter for placeholder/trash bot-safe summaries;
- safe bot-safe metadata in the Wappi draft journal, without raw summary text or raw ids;
- regression coverage for unknown-only, foreign-brand-only, PII-only, trash summary, flag-off, ambiguous identity, and journal metadata.

Original production DB was not modified. A read-only copy was created with SQLite backup API and all probes used that copy.
