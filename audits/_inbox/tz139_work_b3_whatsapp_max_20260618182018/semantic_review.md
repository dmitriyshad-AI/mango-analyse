# Semantic Review

Status: `formal_pass`, semantic review by Codex only; independent Claude regрейд required before acceptance.

Checks:

- WhatsApp messages are stored as timeline history, not as bot-safe context: `allowed_for_bot=False`.
- WhatsApp phone matching does not force-merge ambiguous family/shared phones into one student.
- Ambiguous phone case produces a conflict and manual-review state.
- Brand metadata is carried as `brand:*`/`channel_shared:true`; no cross-brand text merge was introduced.
- MAX events are not invented when no source archive exists.

Known residual risk:

- Real-data dry-run found no ambiguous WhatsApp phone candidates in the selected timeline DB, so the family ambiguous path is validated by fixture tests, not by live WhatsApp sample.
- No semantic review of message content was performed; B3 changes structure and safety only, not customer-facing wording.
