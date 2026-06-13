Risk review

Primary risks addressed:
- CRM losing long objections entirely.
- Auto history exceeding the practical AMO textarea budget and being truncated outside our code.
- Truncation marker being silently erased when an operator needs an explicit signal.
- Evidence compaction cutting words in a visibly broken way.

Guardrails:
- All changed behavior is behind flags.
- The two default-ON flags have explicit OFF NEG tests.
- The default-OFF flag has an unset/default NEG test.
- No live-write path was executed.

Known gaps:
- The tests verify local fixtures and formatting invariants, not production distribution on full ignored snapshots.
- The AMO UI/readback limit is represented by existing constants and unit tests; no live readback was performed.
