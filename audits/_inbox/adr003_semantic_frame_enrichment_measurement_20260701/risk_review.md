# Risk Review

## Live/write risk

Live bot, AMO, Tallanto, CRM and Telegram sends were not touched.

Outputs were written only under `audits/_inbox/`.

## Measurement risks

- Paired enrichment proves a narrow invariant by construction: route/text/safety/checklist are copied from OFF transcript and must remain unchanged. This is correct for metadata no-op, but not enough for decision policy.
- Independent full ON draft runs are nondeterministic and can show route/text diffs even without a behavior bug. They must not be used as strict no-op proof.
- `SemanticFrame` quality remains unproven without manual expected-frame gold.
- Wappi25 is useful but not the full 131-case M1. Full M1 paired enrichment remains a larger acceptance step.

## Guardrails added

- Report now rejects partial enrichment.
- Report now rejects enrichment with non-frame ON calls.
- Report now rejects mismatched input turns.
- Manifest no longer presents same-payload `TELEGRAM_SEMANTIC_FRAME_SHADOW=1` as the canonical Stage 1 command.
