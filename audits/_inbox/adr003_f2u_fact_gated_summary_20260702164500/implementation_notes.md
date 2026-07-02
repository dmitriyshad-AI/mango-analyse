# ADR-003 F2u Implementation Notes

## Scope

Report-only update to `scripts/report_adr003_frame_calibration_queue.py`.

The calibration queue now imports and runs the existing
`report_adr003_fact_gated_self_answer_readiness.py` scorer and exposes its
totals in the combined JSON/Markdown report.

## Why

Fresh M1 `36ea110` showed the real autonomy lever is stable
existence/format proof, not harmless ack/status. The separate fact-gated
report already classified this, but the combined calibration queue did not
surface the numbers.

## Runtime Impact

None.

- No provider/direct-path/profile changes.
- No route/text changes.
- No live bot restart.
- No AMO/Tallanto/CRM/Telegram writes.

## 36ea110 Summary

- strict draft candidates: 0
- manager-only exact-proof rows: 2
- already-self exact-proof rows: 6
- blocked no exact proof: 1
- excluded danger/money/P0: 1

Active remains NO-GO.
