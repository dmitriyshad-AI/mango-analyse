# Risk Review

## Runtime Risk

Low. Only report code and tests changed. No runtime/provider/direct_path code changed.

## Semantic Risk

The main risk is treating a renderer candidate as a sendable customer answer. This is explicitly blocked:

- `active_behavior_allowed=false`;
- full candidate text is not exported;
- candidates require semantic review before any runtime use.

## Business Risk

Current real data has zero renderer candidates. This means the project must not claim autonomy improvement from this phase.

## Data Risk

No AMO/Tallanto/CRM/Telegram writes. No live process touched. No raw `client_safe_text` or `template_text` exported by the new renderer fields.
