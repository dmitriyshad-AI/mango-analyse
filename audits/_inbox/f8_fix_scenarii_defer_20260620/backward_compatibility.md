# Backward compatibility — F8 clean defer

Default behavior outside `pilot_gold_v1` is preserved unless `TELEGRAM_PRICE_AXES_SELECTOR=1` and `TELEGRAM_PRICE_AXES_CLEAN_DEFER=1`.

Inside `pilot_gold_v1`, both F8 flags are now profile-default-on:

- `TELEGRAM_PRICE_AXES_SELECTOR`
- `TELEGRAM_PRICE_AXES_CLEAN_DEFER`

Explicit env value `0` overrides the profile and disables each flag.

When the selector is enabled and an exact price exists, behavior is unchanged: the virtual price fact is still returned first.

When the selector is enabled and no exact price exists:

- without `TELEGRAM_PRICE_AXES_CLEAN_DEFER`: previous fallback fact selection remains available;
- with `TELEGRAM_PRICE_AXES_CLEAN_DEFER`: no unrelated facts are returned for the dead-end price query.
