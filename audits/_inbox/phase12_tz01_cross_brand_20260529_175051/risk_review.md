# Risk review

Risk level: low for this commit, because it only adds one content-safe template and reuses the existing hard re-verification.

Checked risks:
- Double rewrite: dispatcher stops after the first matching template.
- Brand leak: cross-brand output is generic and re-verified.
- False positive on current-brand confirmation: covered by regression.
- False positive on `МФТИ` for UNPK: covered by regression.
- Existing legacy path: unchanged.

Known uncovered risk:
- Later Phase 12 templates may conflict with `cross_brand`; they must be added through the registry with explicit priority tests.

