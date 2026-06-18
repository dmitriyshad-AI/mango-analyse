# Semantic Review

Status: `semantic_pass_with_notes`

## Reviewed Risks

- Family phone must not silently merge different Tallanto students.
- Phone union should work for the same person across source systems and across separate apply runs.
- Brand must not block identity resolution.
- Foton/UNPK must not be mixed into generated client text.
- Mapping must make identity changes reviewable and reversible.

## Findings And Fixes

Independent review initially found three failures:

1. Cross-run generic AMO/Mango phone union only looked at current batches.
   Fix: resolver now reads existing identity links from the native timeline store on apply.

2. Canonical family phone duplicated Mango calls but dedupe kept only one event.
   Fix: family-phone calls use customer-scoped `source_id` and `ambiguous` match status.

3. Canonical split mapping used email in legacy old id.
   Fix: legacy split old id is phone-level only; test now uses different emails across split rows.

Follow-up review confirmed points 1 and 2 and caught point 3; point 3 was fixed and covered by regression.

## Semantic Result

- Family-phone cases are preserved as manager-review conflicts.
- Ordinary same-person phone union does not create extra customers across separate imports.
- Brand is stored as history (`brands`) and does not block identity.
- Work A does not generate customer-facing text, so no new Foton/UNPK mixed output surface was added.

## Residual Notes

- Existing normalizers still persist scrubbed source-like `record` structures; Work A did not expand that surface. A stricter raw-payload semantic gate is a later hardening item.
- Architect should still regrade on real snapshots because this work only used fixtures, per Дмитрий's instruction.
