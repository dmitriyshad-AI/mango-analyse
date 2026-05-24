# Status Rules

## Candidate Status

- `gold_candidate` - may improve examples or tests after review.
- `manager_draft_only` - may only help manager-facing drafts.
- `style_reference_only` - tone/style evidence, not a fact.
- `needs_rop_approval` - business fact needs owner or ROP confirmation.
- `needs_primary_source` - requires source document before any KB change.
- `reject` - unsafe, stale, personal, cross-brand or not useful.

## Fact Sync Status

- `already_confirmed_in_kb`
- `new_dmitry_approved_fact`
- `conflicts_with_current_kb`
- `style_only_not_fact`
- `manager_only_or_needs_check`

## Allowed Transitions

- `new -> needs_rop_approval`
- `new -> needs_primary_source`
- `new -> style_reference_only`
- `new -> manager_draft_only`
- `new -> reject`
- `needs_rop_approval -> approved_for_next_kb_task`
- `needs_primary_source -> approved_for_next_kb_task`
- `needs_rop_approval -> reject`
- `needs_primary_source -> reject`

No status transition writes to KB automatically.
