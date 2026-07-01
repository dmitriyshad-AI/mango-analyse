# Backward Compatibility

## Runtime

No runtime behavior changes.

- No route changes.
- No text changes.
- No new flags.
- No profile/default-on changes.
- No live bot restart.

## Tests And Guards

ADR-003 regex moratorium guard remains unchanged and passed in the targeted test set.

## Data

The committed gold file contains only ids, turns, expected labels, and short non-personal notes. It does not include client message text, phones, emails, raw Wappi payloads, or transcripts.
