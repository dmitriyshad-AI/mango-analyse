# Backward Compatibility

The runtime bot context contract is unchanged:

- `bot_safe_summary` chunk type remains the only visible customer timeline chunk.
- Runtime still reads through `CustomerTimelineReadApi.bot_context(..., allowed_only=True)`.
- Brand filtering remains in `bot_safe_runtime_context.py`.

Behavioral changes:

- `interest/title` now masks person names before rendering the bot-safe summary.
- Stale bot-safe chunks no longer visible to the bot after brand re-resolution; they are retained in storage with `allowed_for_bot=0` and `requires_manager_review=1`.
- Report now includes `retired_stale`.

No live AMO/Tallanto/CRM writes were added.
