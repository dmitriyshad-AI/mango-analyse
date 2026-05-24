# Risk Review

## Main Risks

- Dirty working tree has many unrelated deletions and modified generated KB artifacts.
- Current v6.3 promoted generated artifacts were already dirty before D4.
- The canonical P0 spec exists under `src/mango_mvp/channels/` but is untracked and outside D4 write scope.

## Mitigations

- D4 boundaries documented before implementation.
- No old release deletions were staged or edited.
- No `channels/` files were edited.
- Trial build and Stage6 outputs were written under ignored `audits/_inbox/`.
- Stage6 regression found 1 UNPK brand guard issue; code was fixed and UNPK smoke rerun to zero brand violations.
