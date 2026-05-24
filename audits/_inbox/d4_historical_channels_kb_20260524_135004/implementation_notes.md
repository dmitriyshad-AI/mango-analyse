# D4 Historical Channels KB Implementation Notes

Date: 2026-05-24

## Scope

Implemented D4 KB hygiene without touching `channels/`, `stable_runtime`, AMO, Tallanto, CRM, Telegram live sends, ASR or Resolve+Analyze.

## Changes

- Added D4 file boundaries in `docs/D4_HISTORICAL_CHANNELS_KB_BOUNDARIES_2026-05-24.md`.
- Reworked `scripts/build_kb_release_v6_1_team_answers.py` so business facts come from `facts/*.yaml` plus `release_manifest.yaml`, not Python patch functions.
- Added `release_manifest.yaml` for builder metadata, control numbers and required YAML paths.
- Removed unconfirmed Foton online-year upper bound from source YAML by keeping only `year: 47250`.
- Made post-filter registry brand-aware: global phrases plus `phrases_by_active_brand`.
- Made distribution packs render brand blocks from `brand_rules.yaml`, not Python constants.
- Hardened Stage6 safety: brand-separation guard and resolved unsupported numeric promises force `manager_only`.
- Added docs-only approval queue for future historical-channel candidates.

## Verification

- `pytest --collect-only`: 1835 tests collected.
- KB/Stage6 targeted suite: 57 passed.
- Trial build from v6.3 sources into this audit pack: `quality_passed=true`, `semantic_pass=true`, blocking findings 0.
- Real Codex Stage6 smoke50 against the D4 hygiene build: FOTON 25 rows, UNPK 25 rows, errors 0, brand violations 0, unsupported numeric promises 0.
