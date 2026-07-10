# ADR-003 F2u Semantic Review

## Verdict

`semantic_pass` for diagnostic/report-only use.

## Checked

- The change does not claim that autonomy can be enabled.
- The report explicitly keeps `active_readiness=no_go`.
- The new fact-gated summary distinguishes:
  - no strict draft candidates;
  - manager-only exact-proof rows that need separate policy/upstream work;
  - already-self exact-proof rows with no route leverage;
  - no-proof and danger/P0 rows.

## Business Meaning

The finding matches Claude #1's correction: current data does not support a
quick ack/status route-only demotion. The next useful work is stable
existence/format proof delivery and policy review, not a route switch.

## Residual Risk

This is still an offline scorer over saved transcripts. It does not prove that
future live traffic has the same distribution.
