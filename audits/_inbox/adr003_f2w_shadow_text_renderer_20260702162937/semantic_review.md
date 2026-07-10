# Semantic Review

Verdict: `PASS_WITH_NOTES`

## What Passed

- The phase does not approve any customer-facing text.
- The renderer exports no full candidate text.
- The real 36ea110 run produced zero shadow renderer candidates, so no autonomous text is ready.
- Wrong-brand, unsupported structured values, and missing template renderer are blocked.

## Blocking Issues For Active Use

- No sendable text candidates exist on the fresh 36ea110 run.
- `client_safe_text` direct quoting remains forbidden.
- Template-based facts require a separate renderer and semantic review.
- Wrong-brand exact proof must be fixed upstream before any autonomy can use it.

## Required Next Action

Do not enable F3. The next step should target either wrong-brand proof alignment or a narrowly scoped template renderer in shadow.
