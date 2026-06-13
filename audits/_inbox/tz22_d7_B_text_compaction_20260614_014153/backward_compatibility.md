Backward compatibility

Expected default changes:
- CRM_OBJECTION_COMPACT defaults ON, so long objections now produce compact text instead of an empty field.
- CRM_AUTO_HISTORY_HARD_LIMIT defaults ON, so Auto history can be shorter than before when the composed text exceeds MAX_AUTO_HISTORY_CHARS.

Byte-for-byte NEG:
- CRM_OBJECTION_COMPACT=0 restores the old empty result for long objections.
- CRM_AUTO_HISTORY_HARD_LIMIT=0 restores the old over-limit Auto history composition.
- CRM_KEEP_TRUNCATION_MARK unset/OFF keeps the old truncation-marker stripping behavior in deal-aware normalize_manager_text.

Unaffected:
- Existing short dictionary objection normalization such as "цена" remains unchanged.
- Contact last-summary and next-step compaction behavior was not changed.
- Structured objections are still not sent to AMO payload by default.
