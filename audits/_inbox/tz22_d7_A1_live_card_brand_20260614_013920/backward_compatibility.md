# Backward Compatibility

## Compatibility Status

Compatible by default.

## Existing Behavior Preserved

- New `active_brand` parameters default to `None`.
- REST phone paths still call `_build_dossier_and_analysis(...)` without brand.
- `CRM_LIVE_CARD_BRAND_FAILCLOSED` default OFF preserves old unbranded live-card behavior.
- Existing `brand_mismatch`, `brand_unknown`, `filial_shd`, and `multiple_contacts` reasons remain.
- Public bot context schema already included `active_brand`; this change only propagates it to live Tallanto.

## Intentional Behavior Change

- Public bot live mode now blocks cross-brand live cards through `brand_mismatch`.
- Emergency fail-closed mode can block unverified brands through `brand_unverified`.

## Not Changed

- Server-mode Tallanto context route.
- Brand scope mapping for Foton/unknown филиалы.
