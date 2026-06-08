# Wave 1 Number Scope And Handoff Verifier

Branch: `codex/wave1-number-scope-handoff`
Base: `main` at `6351de27`
TZ: `D1_audit_backlog/TZ_wave1_number_scope_and_handoff_verifier_2026-06-08.md`

## Read-only Entry Map

### B1: scope-aware number gate

- `src/mango_mvp/channels/dialogue_contract_pipeline.py:4207` - `_free_number_gate_findings`.
- Existing flat grounding source was `fact_surfaces = _free_number_surfaces(" ".join(...facts.values()))`.
- Existing product-number finding is emitted from the same function as `unsupported_product_number`.
- Existing scope machinery reused:
  - `mango_mvp.channels.fact_scope_spec.detect_fact_scopes`
  - `fact_scopes_allowed`
  - `blocked_neighbors_for`
- Existing action mapping reused:
  - `wrong_scope -> downgrade`
  - `unsupported_product_number -> block`

No extra number gate or independent scope layer was added.

### B5: pure handoff verifier gap

- `src/mango_mvp/channels/subscription_llm.py:6735` - `apply_semantic_output_verifier`.
- `src/mango_mvp/channels/subscription_llm.py:6766` - pure handoff skip site.
- `src/mango_mvp/channels/dialogue_contract_pipeline.py:6101` - `_handoff_factual_claim_text`.
- `src/mango_mvp/channels/dialogue_contract_pipeline.py:6123` - `_is_pure_handoff_text`.
- Canonical handoff templates reused for whitelist:
  - `SAFE_FALLBACK_DRAFT_TEXT`
  - `_HUMANE_GENERIC_HANDOFF_TEXTS`
  - dialogue-contract `_GENERIC_HANDOFF_TEXTS`
  - dialogue-contract `_HANDOFF_EXHAUSTED_TEXTS`

No additional semantic detector was added.

## Implementation

### Flag 1

`TELEGRAM_NUMBER_GATE_SCOPE_AWARE`, default OFF.

When ON, `_free_number_gate_findings` checks a number against facts carrying the matching product scope, format, and grade. If the same number exists only in a neighboring or conflicting scope, the gate emits `wrong_scope` and downstream routing uses the existing downgrade action.

Touched lines:

- `src/mango_mvp/channels/dialogue_contract_pipeline.py:656` - flag reader.
- `src/mango_mvp/channels/dialogue_contract_pipeline.py:4207` - scope-aware branch in the existing free-number gate.
- `src/mango_mvp/channels/dialogue_contract_pipeline.py:4289` - scoped fact helper data and matching helpers.

### Flag 2

`TELEGRAM_VERIFIER_HANDOFF_CLAIMS`, default OFF.

When ON, pure-handoff skip is limited to an exact normalized whitelist of canonical handoff templates. A handoff-shaped text with substantive client-facing claim is sent to the existing semantic output verifier.

Touched lines:

- `src/mango_mvp/channels/subscription_llm.py:119` - flag constant.
- `src/mango_mvp/channels/subscription_llm.py:6766` - narrowed pure-handoff skip.
- `src/mango_mvp/channels/subscription_llm.py:6873` - whitelist helpers.

## Tests Added

- B1 OFF parity: flat facts still pass when `TELEGRAM_NUMBER_GATE_SCOPE_AWARE=0`.
- B1 wrong scope: online price used in an offline answer emits `wrong_scope`.
- B1 same scope: normalized same-scope number passes.
- B1 adversarial new number: new product price emits `unsupported_product_number`.
- B1 integration: `wrong_scope` downgrades direct-path answer while preserving manager draft text.
- B5 OFF parity: current pure-handoff skip stays unchanged with flag OFF.
- B5 canonical handoff: whitelisted template still skips verifier.
- B5 handoff with claim: substantive handoff goes through verifier and downgrades.
- B5 hard controls: P0/high-risk and brand gate behavior stay intact.

## Not Done

- No simulator or M1 run was executed by request.
- No merge into `main`.

