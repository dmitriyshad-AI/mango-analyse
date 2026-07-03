# ADR003 deletion manifest

Status: data-only, no deletion in этап T.
Owner decision required before any legacy removal.

| legacy candidate | file:line | rule | replacement trace class | negative cases | status |
|---|---:|---|---|---|---|
| availability promise regex | `src/mango_mvp/channels/subscription_llm_parts/reliable_answerer.py:64` | `_AVAILABILITY_PROMISE_RE` detects promises of places/groups/booking | `sense_seats` | seats-vs-place, place-location, live-availability P0-adjacent | `awaiting_green` |
| availability promise caller | `src/mango_mvp/channels/subscription_llm_parts/reliable_answerer.py:281` | `availability_promise_detected(...)` feeds Step1 guard | `sense_seats` | reliable ON/OFF parity, P0/cross-brand bypass | `awaiting_green` |
| off-topic input regex | `src/mango_mvp/channels/subscription_llm_parts/policy_routing.py:553` | `OFF_TOPIC_INPUT_RE` catches out-of-scope input | `off_topic` | identity/prompt injection, metadata-only intent-plan, brand-safe off-topic | `awaiting_green` |
| off-topic template trigger | `src/mango_mvp/channels/subscription_llm_parts/policy_routing.py:4306` | terminal template off-topic disjunct | `off_topic` | no bypass of `off_topic_metadata_only`, no P0 downgrade | `awaiting_green` |
| grade regex memory | `src/mango_mvp/channels/dialogue_memory.py:1008` | `_GRADE_PATTERNS` extracts grade from client text | `slots_gsf` | price numbers, dates, transitions, multi-child | `awaiting_green` |
| subject regex memory | `src/mango_mvp/channels/dialogue_memory.py:1014` | `_SUBJECT_PATTERNS` extracts subject | `slots_gsf` | unsupported subject asked by client, multi-subject ambiguity | `awaiting_green` |
| child-grade regex memory | `src/mango_mvp/channels/dialogue_memory.py:1044` | child/grade patterns extract slots from text | `slots_gsf` | child names, old grades, duplicate children | `awaiting_green` |
| funnel slot extraction bridge | `src/mango_mvp/channels/dialogue_memory.py:1272` | funnel extractors populate memory slots | `slots_gsf` | no merge of semantic-reading slots into known slots | `awaiting_green` |

Controls before changing any status:

- per-class green paired run with multi-mask turns excluded from deletion credit;
- `semantic_reading_trace` proves replacement class saw the same decision surface;
- P0, brand, payment/refund and manager-only floors unchanged;
- Dmitry explicit “yes #2” for deletion.
