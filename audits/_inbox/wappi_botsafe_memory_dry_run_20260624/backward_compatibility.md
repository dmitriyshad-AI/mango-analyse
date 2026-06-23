# Backward Compatibility

Default-off behavior is preserved: when `TELEGRAM_BOT_SAFE_CRM_CONTEXT` is not enabled, the Wappi context builder does not inject bot-safe Customer Timeline context.

When enabled, behavior changes intentionally:

- `unknown` bot-safe chunks no longer satisfy an active-brand request.
- placeholder/trash summaries are dropped.
- safe metadata is added to dry-run/pending journal entries.

The added journal metadata is additive and does not change draft text generation by itself.
