### FC-20260529-1: p0_latch_context_lock

- Status: fixed
- Verdict: real_bot_issue
- Example: synthetic long autonomous dialogue: one refund P0-like turn followed by 5 neutral schedule/address/recording questions.
- Symptom: one active P0 latch could keep the rest of an autonomous long dialogue in handoff mode.
- Root cause: `_next_p0_latch` returned the previous active latch monotonically unless a manager release event was present; autonomous flows do not emit manager release keys. Historical risk flags also reintroduced stale P0 after release.
- Sibling cases: legal/payment-dispute covered as hard non-release controls; neutral bot-frustration covered as non-P0.
- Durable fix: autonomous release for non-hard latches after 5 consecutive neutral client turns; hard legal/payment-dispute latches remain manager-only.
- Regression: `tests/test_dialogue_memory.py` dedicated latch lifecycle tests.
- Owner/next step: Codex implemented partial; Claude semantic review; full Phase 7-strict later covers slot correction/window risks.
