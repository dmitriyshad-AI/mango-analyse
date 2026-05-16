# Stage 1 Readiness Table

| Area | Status | Evidence | Remaining work |
|---|---|---|---|
| Long polling gate | Ready | `TelegramBotPollingConfig` blocks disabled/kill-switch modes; tests pass. | Live start still requires explicit confirmation and env flags. |
| No client auto-send | Ready | `send_client_message` returns blocked result; tests pass; dry-run shows `client_send_enabled=false`. | Keep disabled for phase 1. |
| Debounce | Ready | Consecutive messages from same client are grouped into one draft; tests pass. | Validate in limited live with 3 rapid messages. |
| Manager inbox | Ready for dry-run | Manager message includes source, client text, topic, message type, context quality, RoP decision, draft, risks and follow-up. | Need Nastya chat id and limited live check. |
| Manager buttons | Ready | Only "Принято", "Нужно исправить", "Только менеджер"; no send-client button. | Add richer feedback later if needed. |
| LLM provider | Ready for controlled pilot | `SubscriptionLlmDraftProvider` uses `codex exec`, no OpenAI API key, timeout/fallback, JSON parser, cache outside stable_runtime. | Enable only with `TELEGRAM_PILOT_LLM_ENABLED=1`. |
| Context contract | Ready for next stage | New `PilotContext` and `PilotContextQuality` include customer identity, AMO/Tallanto/timeline/facts quality and warnings. | Need real context builders for AMO/Tallanto/timeline sources. |
| High-risk routing | Ready in LLM result normalization | Refund, matkap, tax, legal, negative feedback, documents, discounts and payment status force manager-only. | Review final high-risk theme list with Dmitry/ROP. |
| Prompt injection guard | Ready | Client message stays inside `<client_message>` and prompt says it is data, not instructions. | Add adversarial historical examples in Stage 6. |
| Safe schedule wording | Ready | Missing schedule facts produce safe template and 24h manager follow-up. | Connect to real fact freshness source later. |
| Runtime writes | Safe | No stable_runtime, AMO, CRM, Tallanto writes in this block. | None. |
