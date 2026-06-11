# Backward compatibility

Runtime behavior was not changed in wave 1.

Compatibility checks:

- full pytest passed: 3004 passed, 5 skipped;
- focused subscription/dialogue/telegram pytest passed: 921 passed;
- replay against frozen baseline passed: 19 cases, 0 mismatches;
- facade export snapshot passed in monolith mode: 771 exports present;
- move-only self-check passed in monolith mode: 484 top-level functions/classes unchanged.

No imports in `src/`, `scripts/`, or `tests/` were rewritten.
