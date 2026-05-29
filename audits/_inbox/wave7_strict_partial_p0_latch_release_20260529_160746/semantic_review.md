Verdict: PASS_WITH_NOTES

Artifact and audience:
- Bot safety/dialogue state behavior for customer-facing autonomous Telegram replies.

What passed:
- Soft refund-style P0 latch no longer silences the rest of a long autonomous dialogue after the client asks 5 consecutive neutral questions.
- Real hard P0 classes remain sticky: legal threat and payment dispute require manager release.
- Bot-frustration text like "вы не отвечаете нормально" is treated as neutral for this release check and does not itself create a P0 latch.
- Release clears the stale held-state P0 flag as well as the authoritative `DialogueMemory.p0_latch`, preventing downstream prompt/intent layers from seeing an already-released latch.

Blocking issues:
- None found in the covered scope.

Non-blocking risks:
- This is a partial Phase 7-strict fix. It does not solve long-dialog slot correction or context-window mismatch.
- Complaint-class latches are allowed to auto-release after 5 neutral turns because the explicit hard set is only legal/payment-dispute. This matches the current task, but should be rechecked in full Phase 7-strict if business wants complaint to be hard-sticky too.
- The release count is regex/rule-based, not LLM-semantic. It is intentionally conservative around latchable P0 markers.

Missing checks:
- No live/simulator behavior was run by instruction.
- No long 15-20 turn LLM dialogue harness was run.

Required regressions/gates:
- Keep the new `tests/test_dialogue_memory.py` latch lifecycle tests as permanent regressions.
- In full Phase 7-strict, add a long-dialog semantic harness for false P0 -> neutral topic recovery and real P0 -> no premature release.

Recommended next action:
- Hand this partial to Claude for semantic review, then continue waiting for the M1 full eval result. Do not start Wave 1a from this commit alone.
