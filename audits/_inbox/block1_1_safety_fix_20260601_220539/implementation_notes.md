# Block 1.1 Safety Fix

Scope: `D1_audit_backlog/TZ_BLOCK1_1_safety_fix_2026-06-01.md`, Block 1.1 only. Block 2 was not started.

Implemented:
- Added persistent `had_hard_p0_claim` on `DialogueP0Latch`. Autonomous neutral release can clear the active latch, but it does not erase the fact that a real refund/legal/payment/complaint P0 happened earlier.
- Suppressed presale refund autonomy after any prior hard P0 claim. A clean hypothetical refund still answers from the refund fact; a real refund claim remains `manager_only` even after neutral turns.
- Tightened A2.1 informational yield for tax/matkap: verified yield now rejects unbacked concrete anchors, unbacked two-children rules, and document/contract scope substitutions.
- Preserved P0 anti-repeat route: humanity antirepeat may vary P0 text, but hard P0 remains `manager_only`. Benign presale refund is not route-locked by the P0 route lock.

Confirmed code points:
- P0/presale-refund path in `dialogue_contract_pipeline.run_pipeline`.
- `_safe_fallback_text` presale refund branch.
- `_augment_with_presale_refund_policy`.
- `_avoid_repeating_text` / P0 antirepeat path via existing tests.
- A2.1 yield path in `subscription_llm._safe_template_yield_result` and `_safe_template_yield_before_fallback`.

No changes:
- Block 2 / selling launchpad.
- Brand guards, high-risk guard policy, verifier core, warmth, KB.
