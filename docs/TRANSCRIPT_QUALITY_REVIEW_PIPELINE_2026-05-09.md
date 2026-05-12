# Transcript Quality Review Pipeline

Дата: 2026-05-09

## Цель

Сделать product-grade pipeline для спорных transcript-quality кейсов:

```text
disputed tasks
 -> gpt-5.4-mini batch review
 -> validator
 -> gpt-5.5 escalation
 -> Claude audit package
 -> consensus queues
 -> staged apply later, not in this step
```

Этот pipeline не пишет в runtime DB. Он формирует решения и очереди для последующего dry-run/apply.

## Реализованные компоненты

- `src/mango_mvp/quality/transcript_quality_llm_review.py`
  - batch LLM-review;
  - Codex CLI provider;
  - JSON schema enforcement;
  - stratified sampling;
  - workers;
  - dry-run mode;
  - fallback from failed batch to single-item calls;
  - writable temporary `CODEX_HOME` for sandboxed workers.

- `src/mango_mvp/quality/transcript_quality_review_validator.py`
  - validates model JSON;
  - checks decision/confidence/evidence;
  - separates low-risk auto candidates from escalation;
  - blocks high-risk contentful/sales/borderline decisions.

- `src/mango_mvp/quality/transcript_quality_escalation.py`
  - builds GPT-5.5 escalation package;
  - writes runnable command for advanced review.

- `src/mango_mvp/quality/transcript_quality_claude_package.py`
  - builds Claude audit folder;
  - includes tasks, GPT mini decisions, GPT-5.5 decisions and prompt.

- `src/mango_mvp/quality/transcript_quality_consensus.py`
  - merges mini/advanced/Claude decisions;
  - outputs final queues: auto-apply, keep, reanalyze, Claude-required, human-required, blocked.

- `scripts/run_transcript_quality_review_pipeline.py`
  - one-command orchestrator for future product usage.

## Live pilot on 1000 disputed calls

Input:

```text
stable_runtime/transcript_quality_disputed_review_20260509/llm_review_tasks.jsonl
```

Output root:

```text
stable_runtime/transcript_quality_pipeline_live_1000_20260509
```

### GPT-5.4-mini review

Config:

- limit: 1000
- sample: stratified
- batch size: 8
- workers: 6
- model: `gpt-5.4-mini`
- reasoning: `medium`

Result:

- selected tasks: 1000
- reviews written: 1000
- errors: 0
- provider calls: 125
- fallback single calls: 72

Decision counts:

- `force_non_conversation`: 867
- `keep_current_analysis`: 104
- `human_review_required`: 20
- `reanalyze_required`: 9

### Mini validator

Result:

- escalation to GPT-5.5: 772
- low-risk auto-apply candidates: 151
- keep-current-analysis candidates: 77
- invalid reviews: 0

### GPT-5.5 escalation

Config:

- tasks: 772
- batch size: 5
- workers: 4
- model: `gpt-5.5`
- reasoning: `medium`

Result:

- reviews written: 772
- errors: 0
- provider calls: 155

Decision counts:

- `force_non_conversation`: 650
- `human_review_required`: 79
- `keep_current_analysis`: 26
- `reanalyze_required`: 17

### Advanced validator

Result:

- auto-apply candidates: 2
- keep-current-analysis: 7
- reanalyze-required: 15
- human/Claude required: 748
- invalid reviews: 0

### Consensus before Claude

Result on the 1000-call pilot:

- `auto_apply_force_non_conversation`: 153
- `keep_current_analysis`: 84
- `reanalyze_required`: 3
- `claude_audit_required`: 760

Interpretation:

- The pipeline is intentionally conservative.
- Low-risk non-contentful cases can proceed toward staged dry-run/apply.
- Most contentful/sales/service/technical conflicts require Claude audit before any mutation.

## Claude package

Built:

```text
stable_runtime/transcript_quality_pipeline_live_1000_20260509/claude_audit_package
```

Files:

- `claude_audit_items.jsonl`
- `claude_audit_items.csv`
- `CLAUDE_AUDIT_PROMPT.md`
- `README_FOR_CLAUDE.md`

Current package size: 300 items.

## Safety policy

No DB write was made by this pipeline.

Allowed later only after Claude/consensus:

1. dry-run apply;
2. staged apply 50;
3. guardrails report;
4. staged apply larger batch;
5. rebuild downstream ROP/KB/CRM artifacts.

## Validation

Targeted tests:

```text
57 passed
```

Full test suite:

```text
510 passed
```

## Next engineering steps

1. Give Claude `claude_audit_package` and collect Claude JSONL decisions.
2. Run consensus with `--claude-reviews-jsonl`.
3. Build staged apply script for consensus `auto_apply_force_non_conversation` only.
4. Build reanalyze queue for `reanalyze_required`.
5. Run the same pipeline on all 3974 disputed tasks after pilot validation.
6. Integrate this quality pipeline into product appliance processing gates.
