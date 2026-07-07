# ADR003 E3 env matrix

Generated at: `2026-07-03T19:13:00+03:00`

## A. Profile default-on flags

Profile env: `TELEGRAM_DIRECT_PATH_PILOT_CONFIG=pilot_gold_v1`

| flag | leg | expected effect |
|---|---|---|
| `TELEGRAM_DIRECT_PATH` | B and ON | enabled by pilot profile |
| `TELEGRAM_BOT_GOLD_REAL` | B and ON | enabled by pilot profile |
| `TELEGRAM_ANSWERABILITY_SHADOW` | B and ON | enabled by pilot profile |
| `TELEGRAM_DEAL_ACTION_DECISION` | B and ON | enabled by pilot profile |
| `TELEGRAM_DIRECT_PATH_MODEL_P0` | B and ON | enabled by pilot profile |
| `TELEGRAM_INTENT_MODEL_LED` | B and ON | enabled by pilot profile |
| `TELEGRAM_P0_MODEL_CLASSES_V2` | B and ON | enabled by pilot profile |
| `TELEGRAM_DIRECT_P0_TEXT_HYGIENE` | B and ON | enabled by pilot profile |
| `TELEGRAM_SEMANTIC_OUTPUT_VERIFIER` | B and ON | enabled by pilot profile |
| `TELEGRAM_OUTPUT_SANITIZER` | B and ON | enabled by pilot profile |
| `TELEGRAM_ROUTE_RUBRIC` | B and ON | enabled by pilot profile |
| `TELEGRAM_DIRECT_PATH_KNOWN_SLOTS_NEXT_STEP_PROMPT` | B and ON | enabled by pilot profile |
| `TELEGRAM_TONE_CLOSE_DETECT` | B and ON | enabled by pilot profile |
| `TELEGRAM_TONE_RICH_FORMAT` | B and ON | enabled by pilot profile |
| `TELEGRAM_A_RICH_FORMAT` | B and ON | enabled by pilot profile |
| `TELEGRAM_LLM_RETRIEVE` | B and ON | enabled by pilot profile |
| `TELEGRAM_FACT_VENUE_SCOPE` | B and ON | enabled by pilot profile |
| `TELEGRAM_AUTONOMY_SCOPE_PRECISION` | B and ON | enabled by pilot profile |
| `TELEGRAM_NUMBER_GATE_SCOPE_AWARE` | B and ON | enabled by pilot profile |
| `TELEGRAM_VERIFIER_HANDOFF_CLAIMS` | B and ON | enabled by pilot profile |
| `TELEGRAM_TEMPLATE_FROM_KB` | B and ON | enabled by pilot profile |
| `TELEGRAM_PRESALE_SAFETY` | B and ON | enabled by pilot profile |
| `TELEGRAM_PRESALE_PII_MEMORY` | B and ON | enabled by pilot profile |
| `TELEGRAM_PRESALE_VERIFIER_FAILSOFT` | B and ON | enabled by pilot profile |
| `TELEGRAM_PRESALE_META_RU` | B and ON | enabled by pilot profile |
| `TELEGRAM_PRESALE_SOURCE_ID` | B and ON | enabled by pilot profile |
| `TELEGRAM_MEMORY_PROVENANCE` | B and ON | enabled by pilot profile |
| `TELEGRAM_MEMORY_PROVENANCE_COMPACT` | B and ON | enabled by pilot profile |
| `TELEGRAM_PII_RELATION_STOPWORDS` | B and ON | enabled by pilot profile |
| `TELEGRAM_MEMORY_CHILD_ELLIPSIS` | B and ON | enabled by pilot profile |
| `TELEGRAM_PRICE_AXES_SELECTOR` | B and ON | enabled by pilot profile |
| `TELEGRAM_PRICE_AXES_CLEAN_DEFER` | B and ON | enabled by pilot profile |

## B. Production-parity env outside profile

| flag | leg | expected effect |
|---|---|---|
| `TELEGRAM_SEMANTIC_FRAME_SHADOW=1` | B and ON | provide the same inline SemanticFrame payload for readers |

## C. ON-only reading delta

| flag | leg | expected effect | negative controls |
|---|---|---|---|
| `TELEGRAM_SEMANTIC_READING_CLASSES=sense_seats,off_topic,slots_gsf,intent_actions,<target>` | ON only | enable the current profile readers plus one target reader for measurement | P0, brand, metadata-only off-topic, slot leak tests |

## Notes

- `B` and `ON` both use `pilot_gold_v1` and reliable answerer.
- Profile defaults now include `sense_seats,off_topic,slots_gsf,intent_actions`; ON must preserve them when adding a target class.
- `ON` differs from `B` only by the additional target class in `TELEGRAM_SEMANTIC_READING_CLASSES`.
- No profile tuple changes are made by this stage.
- Live bot, P0 floor/preblock and legacy deletion are out of scope.
