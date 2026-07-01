# Semantic Review

Verdict: PASS_WITH_NOTES.

This block does not change customer-facing bot behavior. It enforces the ADR-003 rule that new understanding failures should become eval cases and SemanticFrame calibration, not new regex/keyword layers.

Note: this is not a semantic pass for removing existing regex understanding. Existing legacy regex remains frozen until phased migration and measurement.
