# Phase 12 TZ-01 cross_brand · implementation notes

Status: formal_pass.

Scope:
- Added `SafeTemplateSpec` and `apply_dialogue_contract_v2_template_dispatcher`.
- Inserted the dispatcher into `_apply_dialogue_contract_v2_guard_chain` after the verifier guards and before funnel/route permission.
- Registered only the first Phase 12 template: `cross_brand` with priority 10.
- Kept legacy `_cross_brand_safe_template` behavior unchanged.

Important constraints:
- TZ-08 was not implemented.
- Phase 2 / KB v6.4 was not implemented.
- No LLM calls, no 212 run, no CRM/AMO/Tallanto writes, no live-send.

