# Semantic Review

Verdict: `semantic_pass_with_notes`

## Passed

- No historical-channel fact is imported into KB automatically.
- Approval queue rows are masked, single-brand and have `auto_import_allowed=false`.
- Foton installment terms are not globally blocked by the post-filter.
- UNPK keeps `Долями` and `Т-Банк` blocked in the active-brand phrase list.
- Foton online-year client source no longer carries the unconfirmed upper bound.
- P0 list is not duplicated in D4; it references the main dialogue artifact.

## Notes

- `src/mango_mvp/channels/p0_recall_spec.py` is still untracked and belongs to the main dialogue. D4 did not edit it.
- Current promoted v6.3 generated artifacts in `product_data/knowledge_base/...v6_3...` were not overwritten because the working tree already had parallel changes. The checked build was generated in this audit pack from v6.3 sources.
