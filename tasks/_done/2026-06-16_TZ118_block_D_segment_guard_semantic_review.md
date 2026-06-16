# Semantic Review

Дата: 2026-06-16

Артефакт: TZ-118 Block D segment guard shadow measurement.

Аудитория: internal analyst / CRM quality owner.

## Verdict

`BLOCKED`

## Statuses

- formal_pass: yes, tests pass and artifacts were produced.
- semantic_pass: no.
- pilot_ready: no.
- production_ready: no.

## What Passed

- Default mode stays `off`.
- Shadow measurement used Codex CLI only.
- No CRM/Tallanto/AMO/stable_runtime writes.
- Trace includes raw Codex role, post-guard role, guard reason, and fixed/broke effect.

## Blocking Issues

| Priority | Issue | Evidence | Impact | Required Fix |
|---|---|---|---|---|
| P1 | Segment guard repairs to the weak rule and breaks correct Codex turns. | Raw Codex errors: 55/924; post-guard errors: 220/924; guard fixed 17 and broke 182; net -165. | Promoting this D guard would materially worsen mono-call role attribution. | Do not enable primary. Redesign the guard so it does not hand long runs to the current rule engine. |
| P1 | Short service-turn repair is also unsafe on this gold set. | Variant probe: `low_info_plus_anchor` gives 67 errors vs 55 raw, net -12. | Even a smaller deterministic repair would regress quality. | Keep short-turn logic as trace/mark-only until a better gold-backed rule exists. |

## Non-Blocking Risks

- The local design file `2026-06-15_RAZBOR_i_dizayn_gr1_gr4.md` was not found; implementation followed the chat specification.
- Gold set is 23 calls / 924 turns, not a larger acceptance set. The negative signal is strong enough to block primary, but not enough to design the replacement alone.

## Missing Checks

- No larger gold50 acceptance run was available locally.
- No human review of the 182 `broke` turns was performed in this step.

## Regression Tests Or Gate Rules To Add

- Keep automated stop conditions: `post_guard_worse_than_raw`, `guard_net_delta_negative`, `guard_broke_total_positive`.
- Keep raw-vs-post role fields mandatory for D reports.
- Add a semantic gate: any D primary candidate must show non-negative guard net delta and no material increase in segment error rate on gold.

## Version Diff Notes

- Added a new default-off diagnostic guard mode.
- Added before/after tracing for D role assignment.
- No primary behavior was enabled.

## Recommended Next Action

Stop for Claude/Dmitry regrede. Treat the current result as evidence that the proposed deterministic repair is unsafe; next iteration should be mark-only or use a stronger deterministic signal than the current rule roles.
