# Semantic Review

Verdict: PASS_WITH_NOTES

Artifact and audience:
- customer-facing Telegram draft path;
- manager-facing checklist/metadata in transcripts and review queue;
- runner metrics used by Дмитрий/Claude #1 for regrade.

What passed:
- Client text is not enriched with verifier notes; annotations go to metadata/manager checklist only.
- `downgrade_keep_text` preserves the suspect draft for manager review and does not replace it with a generic client fallback.
- Deterministic gate remains final and still blocks brand, P0, meta, promise and number classes after semantic verifier says `ok`.
- Old diagnosis true/false controls are covered in the new verifier path: substantive `manager_only` diagnosis is checked, pure P0/high-risk deferral is skipped, hedged transfer is not downgraded.
- Findings carry `relation_to_base` and nearest fact key, so manager can distinguish contradiction, absent fact and adjacent fact.

Blocking issues:
- None found in code-level semantic review.

Non-blocking risks:
- Live model quality is not proven by unit tests; M1 regrade must verify that the classifier catches the five real semantic classes without excessive downgrades.
- `annotate` manager note is deliberately compressed to two examples; full fidelity depends on reading metadata/summary.
- Future autonomous send mode still needs a separate decision on fail-closed behavior when verifier is unavailable.

Missing checks:
- Live 13+25+hp_topic_change runs on both brains were not run locally; this is explicitly assigned to M1 after bundle.
- No real latency measurement was run; expected extra call is visible in `llm_calls`, but wall-clock needs live data.

Regression tests added:
- Five regrade semantic cases with expected action.
- False controls and prompt controls.
- Deterministic gate anti-unblock.
- Fail-soft timeout retry.
- Diagnosis any-route, hedged false case, pure P0 skip.
- Regen once + full gate with context.
- Autonomous route prohibits regen.
- Cross-model fake replay.
- Runner llm call roles and downgrade-budget dedupe.
