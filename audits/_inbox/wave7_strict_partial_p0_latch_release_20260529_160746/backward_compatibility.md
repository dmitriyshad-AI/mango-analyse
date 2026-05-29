# Backward Compatibility

Preserved:
- Existing manager release keys keep priority: `manager_clear_p0_latch`, `manager_resolved_p0`, `manager_took_over`, `p0_latch_release_event`.
- Existing latch creation mapping is unchanged.
- Existing hard P0 latches for legal/payment dispute remain sticky until manager release.
- Existing public `DialogueMemory` schema is unchanged.

Changed:
- Non-hard active P0 latches can auto-release after 5 consecutive neutral client turns.
- On any latch release, stale historical P0 flags are not carried forward from earlier client turns; current-turn risk flags still apply.
- `held_state.p0_latched` is reset locally when the authoritative dialogue latch is released.

Compatibility tests:
- 358 neighboring bot/safety tests passed.
- 10 semantic role tests passed.
