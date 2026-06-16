# Risk review

## Safety boundaries

- Live AMO/Tallanto writes: not touched.
- ASR / Resolve+Analyze: not run.
- Stable runtime DB/audio/transcripts: not touched.
- Heavy dynamic simulations: not run.

## Main risks checked

- Default OFF: `TELEGRAM_ASSUMED_SCOPE_GUARD` is not in `DIRECT_PATH_PILOT_PROFILE_DEFAULT_ON_FLAGS`.
- Retriever coupling: `TELEGRAM_RETRIEVER_MODEL_DRIVEN` now requires the new guard to be ON.
- P0: guard returns `skipped_p0_or_risk` and does not rewrite high-risk/P0 result.
- No new manager handoff: guard changes text and metadata only, not route.
- Brand: active brand logic is untouched.
- Claim support: `_claim_supported_by_facts` is untouched.

## Residual risk

Full business metric validation requires the planned OFF->ON dynamic run. Unit/integration tests prove contract behavior, not aggregate pilot quality.
