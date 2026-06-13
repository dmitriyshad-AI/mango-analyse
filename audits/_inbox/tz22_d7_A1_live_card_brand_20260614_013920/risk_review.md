# Risk Review

## Changed Behavior

- Calls that pass `active_brand` now enforce existing `brand_mismatch` logic in live card construction.
- `CRM_LIVE_CARD_BRAND_FAILCLOSED=1` blocks any single-contact live card when channel brand is absent.

## Backward Risk Controls

- `CRM_LIVE_CARD_BRAND_FAILCLOSED` defaults to OFF.
- Existing REST phone flows keep `active_brand=None`.
- `skip_shd` remains checked before fail-closed and brand mismatch.
- `CRM_TALLANTO_MODE=mock` still returns disabled before Tallanto client creation.

## Adversarial / Edge Cases Checked

- `mode=mock` with fail-closed ON -> disabled, no Tallanto client.
- `active_brand="foton"` with `filial="mfti"` -> `brand_mismatch`.
- fail-closed ON with no brand and `filial="mfti"` -> `brand_unverified`.
- fail-closed ON with no brand and `filial="onlajn"` -> `brand_unverified`.
- fail-closed OFF with no brand and `filial="mfti"` -> old `ok` card.
- `filial="shd"` with fail-closed ON -> `filial_shd`.
