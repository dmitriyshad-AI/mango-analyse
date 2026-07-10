# Semantic review

Verdict: `PASS_WITH_NOTES`.

## What passed

- The new prompt instruction asks for a short memory summary, not a customer-facing fact claim.
- The summary is not used as a source of truth for prices, dates, seats, refunds, CRM actions, or live availability.
- The summary does not write `known_slots`, `client_confirmed_slots`, CRM, AMO, Tallanto, or external systems.
- OFF leaves the new summary field absent in the direct-path smoke.
- ON smoke produced a safe, brand-local summary for a simple Foton online physics case.
- Auditor-found public pilot/live-check delivery gap was fixed before commit.
- Auditor-found numeric fail-closed gap was fixed before commit.
- Synthetic sentinel personas cover PII, brand crossing, and P0 adjacency at the scenario level.

## Blocking issues

None for a default-OFF formal/directional commit.

## Non-blocking risks

- The draft long-persona set is synthetic and has not yet passed independent semantic review.
- The local real measurement is intentionally small: one direct-path turn ON and one OFF. It proves the transport path, not business quality across long dialogues.
- A model-produced summary can still be too vague, omit an important constraint, or over-compress context. This must be judged on long-dialogue source transcripts before any profile/live inclusion.

## Missing checks

- Full long-persona ON/OFF semantic comparison with real bot and judge was not completed locally.
- No M1 acceptance run was performed by request; this block is not marked semantic-pass.

## Required follow-up

- Fable/Claude should regreyde the draft long personas and local smoke outputs by source before any acceptance claim.
- If a confirmed semantic bug appears, add either a unit test for filtering/transport or a scenario/gate case before enabling the flag outside explicit experiments.
