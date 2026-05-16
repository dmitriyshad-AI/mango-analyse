# Telegram Pilot Contextual Layer - Implementation Notes

Date: 2026-05-17

Scope completed:

- Added explicit pilot context contract with quality markers.
- Expanded draft prompt from "single phrase to answer" to contextual classification plus draft generation.
- Expanded subscription LLM JSON contract with message type, broad group, alternative themes, group/theme confidence, risk level, context usage and context warnings.
- Added safety guards:
  - low theme confidence below 0.70 forces `manager_only`;
  - high-risk themes force `manager_only`;
  - `non_question`, `context_update`, `wait_for_more`, `manager_only` message types force `manager_only`;
  - identity disclosure still falls back to safe manager-only text.
- Added optional LLM-backed `ChannelPreviewService` that can be injected into Telegram polling runtime.
- Long polling remains safe by default. LLM preview is only enabled by `TELEGRAM_PILOT_LLM_ENABLED=1`.
- Manager inbox now shows message type and context quality.

Important limitations:

- No live Telegram long polling was started.
- No client messages were sent.
- No AMO/CRM/Tallanto writes were performed.
- Historical Telegram export sampling was run in safe local mode.
- The 9 969 contextual analysis was not run.

Stage 6 smoke result:

- Source: `telegram_exports (2)/local_vm_2024-04-01/messages.jsonl`
- Output: `.codex_local/telegram_pilot/eval_packs/20260517_contextual_layer_smoke/`
- Eligible dialogs: 707
- Selected dialogs: 20
- Selected messages: 219
- Safety summary: no network, no live send, no CRM/Tallanto/stable_runtime writes.

Next recommended step:

Manually review the private CSV from the Stage 6 smoke pack, then run LLM draft generation over the selected dialogs only after confirming the cases are appropriate for a pilot check.
