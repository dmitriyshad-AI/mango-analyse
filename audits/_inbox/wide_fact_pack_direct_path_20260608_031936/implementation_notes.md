# Wide Fact Pack / Direct Path Pilot

Base HEAD before changes: `6aae8a5b`.

Implemented:

- Replaced the direct-path narrow context facts with a scoped wide fact pack from the active KB snapshot.
- Added hard client-safe filters: active brand, client-allowed facts only, no internal/forbidden facts, and valid `valid_until`.
- Split prompt facts into exact and adjacent blocks, with metadata for selected category and wide fact keys.
- Added number-scope instruction to the direct-path mission prompt.
- Added non-repeating generic replacement text for direct-path hard gate downgrades.
- Added named pilot config `pilot_gold_v1`: v6.6 snapshot, direct path, real-manager gold examples, and v9 judge defaults for M1 tasks.
- Added runner fail-fast: if direct path is configured but the first 4 completed dialogs have zero direct model calls, abort as `config_invalid`.
- Updated defaults to KB v6.6 and documented the direct-path pilot contract in `CLAUDE.md`.

Not done in this commit:

- No live M1 run was executed locally.
- Judge v9 re-judging of historical runs is not executed here.
- M1 task files depend on the final smoke set path and SHA from the architect.
