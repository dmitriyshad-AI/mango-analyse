# ADR-003 F2u Backward Compatibility

## Behavior

No bot behavior changes.

## CLI/API

`scripts/report_adr003_frame_calibration_queue.py` keeps the same command-line
interface. Output JSON/Markdown gains additional summary fields.

## Tests

Targeted report tests pass. Full pytest was run before commit.
