# Scope

## Include

Docs:

- `docs/SAAS_AND_INSIGHT_ROADMAP_2026-05-07.md`
- `docs/SAAS_PRODUCTIZATION_ARCHITECTURE_2026-05-07.md`
- `docs/SAAS_PRODUCTIZATION_DEVELOPMENT_PLAN_2026-05-09.md`
- `docs/SAAS_PRE_PROCESSING_8_PHASES_AUDIT_2026-05-09.md`
- `docs/SAAS_5_STEP_DEMO_READINESS_EXECUTION_2026-05-09.md`
- `docs/SAAS_APPLIANCE_HARDENING_PACK_AUDIT_2026-05-09.md`
- `docs/CHANNEL_BOT_STRATEGIC_PLAN_2026-05-09.md`
- `docs/CHANNEL_RUNTIME_ARCHITECTURE_2026-05-09.md`
- `docs/AI_WORKFLOW.md`
- `CLAUDE.md`

Code:

- `src/mango_mvp/productization/`
- `src/mango_mvp/channels/`
- `src/mango_mvp/amocrm_runtime/routers/agent.py`
- `src/mango_mvp/amocrm_runtime/agent_runtime.py`
- `src/mango_mvp/amocrm_runtime/agent_models.py`

Scripts:

- `scripts/mango_office_*.py`

Tests:

- `tests/test_productization_*.py`
- `tests/test_channels_*.py`
- `tests/test_agent_runtime.py`
- `tests/test_amo_writeback_guards.py`

## Exclude

- Do not audit transcript quality implementation in depth.
- Do not audit ASR/R+A runtime execution.
- Do not mutate `stable_runtime/`.
- Do not treat processing-dialog files as owned by this audit unless they are direct productization dependencies.

## Known Context

The processing-quality thread is separate. It is improving transcript quality, Stage 15 export safety and bot allowlist safety. This audit should focus on the SaaS/productization/channel surfaces that can advance while processing quality is still being hardened elsewhere.
