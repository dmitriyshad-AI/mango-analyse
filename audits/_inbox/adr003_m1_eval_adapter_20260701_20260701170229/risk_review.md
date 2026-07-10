# Risk Review

## Residual risks

1. `expected_frame` gold is still missing for Wappi latest25. This prevents a true per-field SemanticFrame accuracy verdict.
2. The new `scripted` mode must be used for ADR-003 no-op OFF/ON runs; using `client-mode codex` would introduce client-message noise.
3. Some existing regression source files are persona-only; the generated M1 bundle is the canonical runnable scenario file for this phase.

## Mitigations

- Unit tests verify the generated M1 scenario loads through `load_dynamic_sim_input`.
- Fake smoke verifies the scenario can run through the dynamic simulator without model calls.
- The manifest explicitly states the gold limitation and source breakdown.
- PII grep on generated files was reviewed. Phone-pattern hits are ISO timestamps. Generated ADR-003 JSONL files have `@` and email counts = 0 after sanitization. Wappi personal phones/emails remain masked as `[phone]` / `[email]`; likely FIO triples, two-token Russian name pairs, context single names after "ребенка/ученика/записали", and common Russian given names found in Wappi signatures are masked as `[fio]`.
- Two independent audits found and then rechecked earlier sanitizer gaps: current Wappi message duplication in history and unmasked Wappi name pairs. Both were fixed, regenerated, and covered by tests.
