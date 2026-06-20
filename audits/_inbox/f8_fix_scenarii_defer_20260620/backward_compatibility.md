# Backward compatibility — F8 clean defer

Default behavior is preserved unless `TELEGRAM_PRICE_AXES_CLEAN_DEFER=1`.

`TELEGRAM_PRICE_AXES_SELECTOR` remains default-off and was not added to `pilot_gold_v1`.

When the selector is enabled and an exact price exists, behavior is unchanged: the virtual price fact is still returned first.

When the selector is enabled and no exact price exists:

- without `TELEGRAM_PRICE_AXES_CLEAN_DEFER`: previous fallback fact selection remains available;
- with `TELEGRAM_PRICE_AXES_CLEAN_DEFER`: no unrelated facts are returned for the dead-end price query.
