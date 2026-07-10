# Backward Compatibility

## Runtime

No runtime wiring. Direct path, provider, post layers, profile flags, P0 floor and live bot behavior are unchanged.

## Tests

Existing price axes catalog tests still pass together with the new product existence catalog tests.

## Data

The KB snapshot is read by tests and summary generation only. No KB files are modified.

## API

The new module is additive. No existing public function signatures changed.
