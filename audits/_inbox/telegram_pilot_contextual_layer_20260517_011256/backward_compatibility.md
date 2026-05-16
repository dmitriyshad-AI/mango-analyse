# Backward Compatibility

- Existing deterministic `ChannelPreviewService` behavior remains available and default.
- `TelegramBotPollingRuntime` still defaults to deterministic preview service unless an LLM service is explicitly injected.
- `scripts/telegram_manager_draft_pilot.py` uses LLM preview only when `TELEGRAM_PILOT_LLM_ENABLED=1`.
- Existing `topic_confidence` field remains supported. New `confidence_theme` is accepted as an alias and exported alongside `topic_confidence`.
- Existing fake provider and parse helpers remain compatible.
- Manager inbox still includes all previous sections, with two added sections: message type and context quality.
- No existing runtime artifact paths were changed.
