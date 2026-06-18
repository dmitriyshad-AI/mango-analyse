# Source Discovery

## WhatsApp

Source: `/Users/dmitrijfabarisov/Projects/Mango analyse/all_whatsapp_chats.txt`

Read-only parse stats:

```json
{
  "chats_seen": 4620,
  "unique_chats": 4620,
  "messages_seen": 63584,
  "records_built": 40034,
  "skipped_service": 7416,
  "skipped_empty": 16134,
  "skipped_malformed": 1,
  "linked_by_phone": 39999,
  "session_only": 35,
  "chats_linked_by_phone": 4615,
  "chats_session_only": 5
}
```

Git ignore check in canonical repo:

```text
.gitignore:197:all_whatsapp_chats.txt all_whatsapp_chats.txt
```

`product_data/transcripts/whatsapp_chats.sqlite` was present but had no tables, so B3 used the raw txt export.

## MAX

No MAX message archive was found under checked `product_data` paths. Matches were KB/contact/channel mentions, not a message archive. B3 therefore leaves MAX as blocked/no-source and does not synthesize events.
