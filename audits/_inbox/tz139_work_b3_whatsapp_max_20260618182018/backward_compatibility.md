# Backward Compatibility

- Existing Telegram/MAX/web channel behavior remains mapped to their prior event/link types, except WhatsApp now gets explicit `whatsapp_message`/`whatsapp_user_id` instead of generic web-chat types.
- Existing `web_chat_message` remains available for site/web channels.
- Existing WhatsApp txt importer remains dry-run by default and keeps `--apply` semantics unchanged.
- Existing tests expecting generic WhatsApp web-chat type were updated to the new explicit contract.
- No migrations or existing SQLite files were modified.
