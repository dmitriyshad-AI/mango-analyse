# Risk review

## Main risks

- Tallanto early stop can skip later contacts if the same phone appears in multiple fields and those fields point to distinct contacts.
- AMO batch correctness depends on external amoCRM `filter[id]` contract.
- PROFILE phone index requires a rebuild/write path to populate `primary_phone_norm`.

## Mitigations

- Tallanto OFF path remains available through `TALLANTO_BATCH_FETCH=0`.
- Tallanto test verifies same live card with fewer mock calls.
- AMO official docs confirm `filter[id]` for leads accepts arrays and is multiple.
- AMO mock test verifies URL contract, order preservation, equivalent candidate set, equivalent selected lead, and fewer calls.
- PROFILE tests verify explicit OFF, default ON, column/index creation, same IDs as full scan, and no suffix overmatch in the tested tail case.

## Remaining operational note

No profile rebuild was run. Populate `primary_phone_norm` in the single TZ-32 rebuild, not in a separate TZ-33 rebuild.

