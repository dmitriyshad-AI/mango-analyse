TZ22 D7 block B: CRM text compaction fixes

Scope:
- B1: long structured objections are compacted instead of dropped when CRM_OBJECTION_COMPACT=1.
- B2: contact Auto history is hard-limited to MAX_AUTO_HISTORY_CHARS when CRM_AUTO_HISTORY_HARD_LIMIT=1.
- B3: local deal-aware normalize_manager_text can preserve a single truncation marker when CRM_KEEP_TRUNCATION_MARK=1.
- B4: short evidence and history fallback compaction now cut on word boundaries and keep the required suffix.

Flags and defaults:
- CRM_OBJECTION_COMPACT default: ON.
- CRM_AUTO_HISTORY_HARD_LIMIT default: ON.
- CRM_KEEP_TRUNCATION_MARK default: OFF.

Implementation:
- Added word-boundary helper in deal_text_builder.py for B4 suffix compaction.
- Added compact_objection() with the B1 ellipsis suffix, gated by CRM_OBJECTION_COMPACT.
- Kept old long-objection drop behavior behind CRM_OBJECTION_COMPACT=0.
- Added CRM_KEEP_TRUNCATION_MARK behavior only to deal-aware normalize_manager_text; default still strips markers.
- Applied Auto history hard limit after composing all blocks in write_amo_ready_contacts.py.

Out of scope:
- No live AMO/Tallanto writes.
- No ASR/analyze/heavy reruns.
- No large ignored snapshot measurements; Dmitry explicitly assigned those checks to Claude.
