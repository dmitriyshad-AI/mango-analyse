# Wave 7-strict partial: autonomous P0 latch release

Status: formal_pass.

Scope:
- Implemented only the autonomous lifecycle for `DialogueMemory.p0_latch`.
- Did not change slot correction, context windows, rolling summary, KB, eval sets, CRM/AMO/Tallanto, or live-send behavior.

Code changes:
- `src/mango_mvp/channels/dialogue_memory.py`
  - Added `AUTONOMOUS_P0_LATCH_RELEASE_NEUTRAL_TURNS = 5`.
  - Added autonomous release in `_next_p0_latch` when the previous latch is active, the latch is not hard legal/payment-dispute, and the last 5 client turns contain no latchable P0 markers.
  - Kept manager release events as the first and strongest release path.
  - Kept hard latches (`legal`, `legal_threat`, `payment_dispute`) manager-only.
  - When a latch is released, old historical P0 flags from earlier turns are not reintroduced from the full dialogue history; only current-turn risk flags remain.
  - Cleared `held_state.p0_latched/p0_codes` locally on release so downstream prompt/intent layers do not keep a stale P0 flag.

Regression tests:
- Added first dedicated latch lifecycle regressions in `tests/test_dialogue_memory.py`.
- Covered refund latch auto-release after 5 neutral client turns.
- Covered neutral bot-frustration text not starting or extending P0.
- Covered legal/payment-dispute latches not auto-releasing.
- Covered manager release not keeping stale `p0` / `payment_dispute` flags from history.

Known not covered in this partial:
- Slot correction risk from long-dialog analysis.
- Window alignment / rolling semantic summary.
- Full Phase 7-strict long-dialog harness.
