# ADR003 deletion manifest

Status: first owner-approved deletion pass started on 2026-07-04.
Owner decision "yes #2" received for E3-backed removal; safety floors remain blocked until their replacements are active.

| legacy candidate | file:line | rule | replacement trace class | negative cases | status |
|---|---:|---|---|---|---|
| availability facet markers | `src/mango_mvp/channels/subscription_llm_parts/reliable_answerer.py:32/48/61` | client/fact facet `"availability"` for Step1 coverage planning | `sense_seats` | live availability floor and promise floor stay intact | `removed_2026-07-04` |
| availability promise regex | `src/mango_mvp/channels/subscription_llm_parts/reliable_answerer.py:63` | `_AVAILABILITY_PROMISE_RE` detects promises of places/groups/booking in bot output | output verifier, not client understanding | availability promises without live fact still blocked | `kept_as_output_floor` |
| availability promise caller | `src/mango_mvp/channels/subscription_llm_parts/reliable_answerer.py:278` | `availability_promise_detected(...)` feeds Step1 guard | output verifier, not client understanding | reliable ON/OFF parity, P0/cross-brand bypass | `kept_as_output_floor` |
| off-topic input regex | `src/mango_mvp/channels/subscription_llm_parts/policy_routing.py:553` | `OFF_TOPIC_INPUT_RE` catches out-of-scope input | `off_topic` | identity/prompt injection, metadata-only intent-plan, brand-safe off-topic | `removed_2026-07-04` |
| off-topic template trigger | `src/mango_mvp/channels/subscription_llm_parts/policy_routing.py:4306` | terminal template off-topic disjunct | `off_topic` | no bypass of `off_topic_metadata_only`, no P0 downgrade | `removed_2026-07-04` |
| grade regex memory | `src/mango_mvp/channels/dialogue_memory.py:1019` | `_GRADE_PATTERNS` extracts grade from client text | `slots_gsf` | price numbers, dates, transitions, multi-child | `blocked_until_slots_gsf_known_slots_merge` |
| subject regex memory | `src/mango_mvp/channels/dialogue_memory.py:1025` | `_SUBJECT_PATTERNS` extracts subject | `slots_gsf` | unsupported subject asked by client, multi-subject ambiguity | `blocked_until_slots_gsf_known_slots_merge` |
| format regex memory | `src/mango_mvp/channels/dialogue_memory.py:1034` | `_FORMAT_PATTERNS` extracts format | `slots_gsf` | format choice, address-vs-format, bot/KB quote leakage | `blocked_until_slots_gsf_known_slots_merge` |
| child-grade regex memory | `src/mango_mvp/channels/dialogue_memory.py:1055` | child/grade patterns extract per-child slots from text | `slots_gsf` | child names, old grades, duplicate children | `kept_until_multi_child_semantic_slots` |
| funnel slot extraction bridge | `src/mango_mvp/channels/dialogue_memory.py:1280` | funnel extractors populate memory slots | `slots_gsf` | no merge of semantic-reading slots into known slots | `blocked_until_slots_gsf_known_slots_merge` |

Controls before changing any status:

- per-class green paired run with multi-mask turns excluded from deletion credit;
- `semantic_reading_trace` proves replacement class saw the same decision surface;
- P0, brand, payment/refund and manager-only floors unchanged;
- Dmitry explicit “yes #2” for deletion.
- For slot deletion, `slots_gsf` must first write safe inferred slots into `known_slots` with `source=semantic_reading_llm` and must not populate `client_confirmed_slots`.
