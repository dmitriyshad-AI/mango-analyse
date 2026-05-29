# Risk review

Risk level: medium-low.

Checked risks:
- Wrong-scope olympiad substitution is blocked in v2 for regular online questions.
- Legitimate olympiad-online 9/11 answer remains possible.
- Unsupported numeric guard still runs before dispatcher.
- Topic normalization is explicit and logged with `program_topic_normalized`.

Known uncovered risk:
- If the contract fails to mark `regular_online` / `blocked_neighbor_scopes`, the wrapper may not detect every regular-vs-olympiad substitution. Phase 1 fact-use enforcement remains the next larger fix.

