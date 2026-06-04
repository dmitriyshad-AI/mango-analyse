# Backward Compatibility

## Flag OFF

`TELEGRAM_Q_PARTIAL_YIELD` defaults to OFF. A regression test verifies that the existing cite-only recovery path remains unchanged when the flag is absent.

## Existing Safety Paths

- P0 route remains `manager_only`.
- Refund policy without refund fact remains refund handoff and does not substitute a neighboring price/course fact.
- Foreign-brand text in a retrieved fact is blocked before client output.
- Product price estimate remains blocked even if the text includes travel words.

## Existing Tests

Full pytest passed with the expected v6.4 KB release artifact:

```text
2498 passed, 5 skipped, 1 warning in 41.17s
```

## Known Environment Note

This worktree does not contain `product_data/knowledge_base/kb_release_20260530_v6_4_team_answers`; full KB import tests require `KB_RELEASE_V3_DIR` pointing to the accepted v6.4 artifact in `.phase12`.
