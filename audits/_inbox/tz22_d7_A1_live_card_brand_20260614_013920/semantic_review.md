# Semantic Review

Verdict: PASS_WITH_NOTES

## Artifact And Audience

- Artifact: read-only Tallanto live card context for CRM/public bot.
- Audience: CRM/AMO/Tallanto internal manager context and public Telegram bot context builder.

## What Passed

- Public bot live CRM context now passes channel brand to live Tallanto context.
- Foreign UNPK live card under `brand="foton"` is blocked as `brand_mismatch`.
- Matching UNPK live card under `brand="unpk"` remains available.
- Fail-closed flag can block unverified brand as `brand_unverified`.
- Default fail-closed OFF keeps old behavior for unbranded calls.
- `mode=mock` still exits as disabled before Tallanto client creation.

## Blocking Issues

- None for the live local path covered by A1.

## Non-Blocking Risks

- Server-mode Tallanto route does not accept `active_brand` and was not changed in this block.
- Unknown/Foton filial mapping remains unchanged by design; expanding it requires separate source-backed mapping work.
- Raw live Tallanto data was not queried; all checks are mocked/unit-level.

## Required Regression Tests / Gate Rules

- Public bot live context with `brand="foton"` and UNPK filial -> `brand_mismatch`.
- Public bot live context with `brand="unpk"` and UNPK filial -> card `ok`.
- `CRM_LIVE_CARD_BRAND_FAILCLOSED=1` with no brand -> `brand_unverified`.
- Default `CRM_LIVE_CARD_BRAND_FAILCLOSED` OFF with no brand -> old card behavior.

## Acceptance Note

Blocker before enabling `CRM_TALLANTO_MODE=http` in production: public bot `active_brand` pass-through is now implemented and tested for live mode. Because fail-closed is default OFF by Дмитрий's decision, this pass-through is the active brand-isolation control.
