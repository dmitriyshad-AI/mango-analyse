Semantic review for CRM text layer block B

Checks:
- Long objections no longer disappear by default; they remain visible as a compact phrase with a visible ellipsis.
- Auto history now has a deterministic hard ceiling and preserves the existing "[сжато]" marker instead of silent clipping or UI-side truncation.
- Default CRM_KEEP_TRUNCATION_MARK=OFF preserves old text cleanup behavior.
- When CRM_KEEP_TRUNCATION_MARK=ON, duplicate truncation markers collapse to one explicit "[сжато]" at the end.
- B1 and B4 intentionally use different suffixes: B1 uses "…" per TZ; B4 keeps "[сжато]" or ". Детали в полном звонке." per existing CRM copy.

NEG covered:
- CRM_OBJECTION_COMPACT=0 keeps old long-objection drop behavior.
- CRM_AUTO_HISTORY_HARD_LIMIT=0 keeps old long Auto history behavior.
- CRM_KEEP_TRUNCATION_MARK unset keeps old marker-stripping behavior.

Residual semantic risk:
- This is a fixture-level semantic review. Large snapshot checks for real slot deltas and sample rows were not run in this worktree because the ignored source snapshots are absent and Dmitry assigned those checks to Claude.
