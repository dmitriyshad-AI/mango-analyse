# Risk review

Residual risk accepted by TZ30:

- If the model merges two different children with compatible grades and no invented names, no automatic name-based guard remains.
- The impact is a merged child slot in the local profile card, not AMO/Tallanto write or brand leakage.
- This is caught by manual regread over raw local diagnostics, not by string matching.

Known limitation:

- If extraction puts two siblings into one mention such as `Артём, Марк`, the resolver still assigns that mention to one model child. TZ30 does not fix this extraction defect.

Safety boundaries:

- No AMO/Tallanto/CRM writes.
- No ASR.
- No Resolve+Analyze full run.
- No full 7512 profile run.
- Microprobe wrote only ignored local files under `product_data/customer_profiles/tz30_microprobe_v4_20260614_142934`.
- Raw names are present only in local ignored diagnostics and SQLite/cache artifacts; audit pack contains only aggregates and paths.
