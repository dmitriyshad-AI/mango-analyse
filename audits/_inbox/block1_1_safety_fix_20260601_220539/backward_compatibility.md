# Backward Compatibility

Expected preserved behavior:
- Existing presale refund multiturn followups still answer from the refund fact when there was no prior hard P0 claim.
- Autonomous release still clears the active refund latch after five neutral turns, but now keeps historical hard-P0 state.
- Existing tax safe template can still yield to verified fact answer.
- Existing complaint/refund templates and anti-repeat variants stay in the manager route.

Compatibility evidence:
- Touched module suite: 488 passed.
- Targeted Block 1.1 regressions: 7 passed.
- Local fake smoke: ok=true, errors=0.

Not covered by this block:
- Block 2 selling launchpad.
- KB changes.
- Live Telegram send.
- AMO/CRM/Tallanto writes.
