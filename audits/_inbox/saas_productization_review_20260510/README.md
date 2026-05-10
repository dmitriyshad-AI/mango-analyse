# SaaS Productization Review Pack

Date: 2026-05-10

Purpose: independent Claude Code audit of the SaaS/productization and channel/bot foundation work.

This is a narrow audit pack. It is not a processing-quality audit and must not expand into transcript-processing fixes.

## Audit Goal

Check whether the current SaaS/productization layers are safe and coherent enough to continue toward client-hosted SaaS:

- Mango capture / shadow polling / recording capture planning;
- product DB and product API boundaries;
- scheduler / appliance / operations hardening;
- CRM and Tallanto read-only snapshot / preview / writeback guardrails;
- dashboard/API data contracts;
- tenant/client-hosted model;
- channel-neutral bot foundation;
- Telegram/site/CRM chat read-only adapters;
- approval workspace primitives;
- revenue feedback loop.

## Hard Safety Boundaries

Claude Code must not:

- edit anything outside `audits/_results/`;
- write to AMO/CRM/Tallanto;
- run ASR;
- run R+A;
- run heavy batch/start/run-ui scripts;
- mutate `stable_runtime/`;
- delete files;
- change implementation files or tests.

## Expected Output

Write results to:

```text
audits/_results/2026-05-10_saas_productization_review/
```

Required files:

- `CLAUDE_REAUDIT_RESULT.md`
- `findings.csv`
- `row_decisions.csv`

Final verdict:

- `PASS`
- `PASS_WITH_LIMITATIONS`
- `FAIL`

## Primary Questions

1. Are there hidden live-write paths to AMO/CRM/Tallanto where the docs claim read-only/dry-run?
2. Are product DB and runtime DB boundaries clear enough?
3. Are approval gates and no-live-send guarantees coherent across channel adapters?
4. Are scheduler/appliance/client-hosted assumptions realistic?
5. Are tests sufficient for idempotency, no live send, no CRM write and data persistence boundaries?
6. What are the highest-priority fixes before persistent channel storage / read-only API / UI approval workspace?
