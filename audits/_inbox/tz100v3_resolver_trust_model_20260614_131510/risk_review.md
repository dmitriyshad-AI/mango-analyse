# Risk review

Primary risk: false trust in `merge_confidence`.

Observed result: on known-bad families, the model still emits `high` for bad or vetoed merges. Therefore confidence is logged only and must not weaken name-veto logic.

Data safety:

- No AMO/Tallanto/CRM writes.
- No ASR.
- No Resolve+Analyze full run.
- No full 7512 profile run.
- Microprobe wrote only local ignored files under `product_data/customer_profiles/tz100_microprobe_v3_20260614_124341`.
- Raw names are present only in `name_veto_diagnostics.local.jsonl`, an ignored local artifact for manual classification.
- Audit pack does not include raw names, phones, cache, SQLite DBs, or raw diagnostics.

Residual risk:

- Manual classification is still required to separate real model errors from false ASR/name-variant rejections.
- One known-bad case accepted under v3 prompt; this needs manual review from raw local diagnostic/trace context before any behavior decision.
- OpenAI provider still uses `json_object`; strict schema is enforced for Codex CLI and by local normalize/validate for all providers.
