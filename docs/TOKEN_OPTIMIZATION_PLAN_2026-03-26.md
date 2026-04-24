# Token Optimization Plan (2026-03-26)

## Goal
Reduce token consumption across `Transcribe` merge, `Resolve`, and `Analyze` without measurable quality regression in production outputs.

## Current findings

### 1. Analyze is the main recurring LLM cost center
Measured on local dataset:
- `Analyze` prompt size, `compact` profile, 1000 ready calls:
  - mean prompt size: `3928.6` chars
  - median: `3302`
  - p90: `6444`
  - p95: `8332`
- `Analyze` prompt size, `full` profile:
  - mean: `4610.1` chars
  - median: `3911`
  - p90: `6968`
  - p95: `8895`
- Transcript size on 2000 ready calls:
  - mean: `1420.4` chars / `238.6` words
  - median: `743` chars / `120` words
  - p95: `5274` chars / `936` words
- Deterministic hints size:
  - mean: `406.9` chars
  - median: `399`

Interpretation:
- Hints are not the main problem.
- Main cost is transcript + verbose schema/prompt + repeated summary generation.

### 2. Compact -> full escalation is already rare
On 1000 analyzed calls, current escalation heuristic would trigger `full` only for:
- `46 / 1000` calls (`4.6%`)

Interpretation:
- Optimizing `full` still matters, but it is not the biggest lever.
- Most savings must come from the `compact` path and from reducing total number of LLM calls.

### 3. Stage-1 transcribe merge hides a large amount of Codex usage
Sampled on 2000 successfully transcribed calls:
- `merge_provider = codex_cli`: `1582`
- `merge_provider = primary`: `203`
- `merge_provider = rule`: `215`

Interpretation:
- The project currently spends many LLM calls before `Resolve` and `Analyze` even start.
- This is likely one of the biggest hidden token drains.

### 4. Resolve LLM usage is already selective, and its marginal value is low
Measured on completed resolve records:
- `accept_baseline`: `2586`
- `accept_rescue`: `322`
- `accept_llm`: `3`
- `resolve_has_llm`: `213`
- `resolve_has_rescue`: `327`

Interpretation:
- `Resolve` LLM is not the main quality driver.
- Rescue ASR matters more than LLM in current architecture.
- Extra LLM use in `Resolve` should be even more tightly gated.

## Principles for token reduction without quality loss

### A. Remove work the model should not do at all
If a field can be deterministically derived in code, do not ask the model for it.

### B. Avoid repeated LLM work on identical inputs
Use content-addressed caching by prompt version + model + normalized transcript hash.

### C. Route easy calls away from LLM
Use deterministic fast paths for clearly non-sales/support/technical buckets.

### D. Spend expensive LLM only where ambiguity exists
Reserve high-cost passes for calls with actual uncertainty or high business value.

## Recommended plan

## Phase 1. Zero-regression structural savings
These changes should not reduce quality if implemented carefully.

### 1. Stop asking Analyze LLM for legacy fields
Current `Analyze` normalization already derives and rewrites many fields.
The LLM should return only the minimal semantic core.

#### Keep in LLM output
- `history_summary_core` or `history_summary`
- `people`
- `contacts`
- `student`
- `interests`
- `commercial`
- `objections`
- `next_step`
- `lead_priority`
- `target_product` only if it cannot be deterministically mapped from `interests.products`

#### Remove from LLM output
- `history_short`
- `crm_blocks`
- `summary`
- `interests` (legacy flat array)
- `student_grade`
- `personal_offer`
- `pain_points`
- `budget` (duplicate)
- `timeline` (duplicate)
- `next_step` duplicate top-level string
- `follow_up_score`
- `follow_up_reason`
- `tags`
- `quality_flags`
- `needs_review`
- `review_reasons`
- `evidence`

#### Generate deterministically in code
- `history_short`
- `crm_blocks`
- `summary`
- `student_grade`
- `interests` flat list
- `follow_up_score`
- `follow_up_reason`
- `tags`
- `quality_flags`
- `needs_review`
- `review_reasons`
- `evidence`

Expected benefit:
- lower output token volume
- lower hallucination surface
- more stable schema

### 2. Move summary framing into code
Current prompt requires the model to include call date/time and manager name in sentence 1.
That is deterministic metadata and should be composed in code.

#### New approach
- LLM returns only a short factual `summary_core` without date/manager preamble.
- Python composes final `history_summary` from:
  - deterministic intro (`started_at`, `manager_name`)
  - `summary_core`
  - normalized next step if needed

Expected benefit:
- shorter output
- fewer inconsistent opening sentences
- more stable summary style

### 3. Enforce compact minified JSON output
Prompt should explicitly require:
- single-line JSON
- no indentation
- no explanatory text

Expected benefit:
- small but free output token reduction
- easier parsing

### 4. Add response cache for Analyze and Resolve
Cache key:
- stage (`analyze`, `resolve_dialogue`, `merge_pair`)
- model
- reasoning
- prompt_profile
- prompt_version
- normalized input hash

Cache payload:
- exact parsed JSON response
- timestamp
- validation status

Expected benefit:
- avoids re-spending tokens on retries, migrations, A/B reruns, repeated exports
- especially valuable in this project because re-analysis and benchmark loops are frequent

## Phase 2. Reduce number of Analyze LLM calls
These changes can preserve quality if gated conservatively.

### 5. Add deterministic fast path for obvious non-sales calls
Applicable buckets:
- `non_conversation`
- `technical_call`
- `service_call`
- `existing_client_progress`

Initial safe rule:
- only skip LLM when heuristic classification is high-confidence and there is no strong sales signal
- generate summary from templates + extracted entities
- mark `needs_review=true` only when heuristic confidence is not enough

Observed room:
- among first 1000 analyzed calls:
  - `service_call = 175`
  - `technical_call = 70`
  - `existing_client_progress = 41`
  - `non_conversation = 159`
- total potentially non-sales buckets: `445 / 1000`

Practical safe target:
- first aim to bypass LLM for `15-25%` of calls only
- do not aim at full `44.5%` immediately

Expected benefit:
- largest cost reduction in `Analyze`
- zero token usage on many administrative calls

### 6. Add transcript compaction before Analyze LLM
Deterministic compaction should remove only low-information content:
- repeated greetings
- repeated acknowledgements (`да`, `угу`, `спасибо`, `хорошо`) in long runs
- exact duplicate lines
- obvious boilerplate
- repeated contact/service scaffolding already available in metadata

Important:
- preserve all product, subject, objection, schedule, agreement, and next-step content
- compaction must be reversible for audit: keep original transcript untouched in DB, compact only for prompt input

Expected benefit:
- lower input token volume on long calls
- likely `10-25%` input reduction on average, more on service calls

### 7. Keep compact as default, but tighten full escalation inputs
Current escalation rate is only `4.6%`, which is acceptable.
However, the full prompt is still too verbose.

Do:
- keep escalation
- shrink `full` schema too
- reuse deterministic derivations there as well

Expected benefit:
- preserves safety net
- lowers cost of ambiguous cases too

## Phase 3. Reduce hidden LLM cost before Analyze

### 8. Rework stage-1 Codex merge gating
Current sampled distribution shows `codex_cli` merge on `1582 / 2000` transcribed calls.
This is likely too broad.

Recommended change:
- keep rule/primary merge by default
- call Codex merge only when at least one is true:
  - low similarity and large semantic divergence
  - high warning density in pair merge
  - empty/weak baseline after rule merge
  - risky high-value call

Do not change blindly.
Benchmark required:
- `100` same calls
- compare current `codex_cli` merge vs gated/rule-first merge
- compare downstream `Resolve` + `Analyze` outputs

Expected benefit:
- potentially massive token reduction
- but must be benchmarked carefully because this stage affects transcript quality

### 9. Make Resolve LLM even more selective
Current results show:
- LLM touched `213` calls
- final `accept_llm` happened only `3` times

Recommended change:
- default to baseline + rescue
- run dialogue-level LLM only when:
  - same-ts disorder is severe
  - rescue is unavailable or weak
  - call is high-value and still ambiguous

Expected benefit:
- fewer expensive dialogue prompts
- negligible quality loss if gating is correct

## Phase 4. Operational controls

### 10. Add token/call accounting to stats and UI
Expose per stage:
- LLM calls count
- cache hits / misses
- escalations to full
- fast-path bypass count
- average compacted transcript length
- estimated token budget consumed

This is required for production control.
Without visibility, token regressions will go unnoticed.

### 11. Version prompts explicitly
Add version labels such as:
- `analyze_prompt_version = v3`
- `resolve_dialogue_prompt_version = v2`
- `merge_pair_prompt_version = v2`

This is required for:
- cache correctness
- benchmark correctness
- safe rollback

## Priority order
1. Remove legacy output fields from Analyze LLM and generate them in code.
2. Move summary metadata framing into code.
3. Add minified JSON response requirement.
4. Add response cache.
5. Add conservative fast path for obvious non-sales calls.
6. Add transcript compaction.
7. Shrink full prompt too.
8. Benchmark and narrow stage-1 Codex merge usage.
9. Tighten Resolve LLM gating.
10. Add token counters to stats/UI.

## Expected impact
If implemented carefully:
- immediate `Analyze` token reduction without quality regression from schema slimming alone
- medium-term major savings from fast-path + cache
- potentially largest total savings from reducing stage-1 Codex merge overuse

## Non-recommended ideas
These are not recommended for a quality-critical production system:
- batching multiple unrelated calls into one LLM request
- blindly switching all analysis to a cheaper model without A/B
- removing escalation safeguards
- dropping rescue ASR before proving transcript quality is unaffected
