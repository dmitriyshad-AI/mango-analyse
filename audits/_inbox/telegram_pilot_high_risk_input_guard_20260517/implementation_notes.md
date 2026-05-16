# Telegram Pilot High-Risk Input Guard

Date: 2026-05-17

Problem confirmed:

- Dialog `1063099421` was about refund, but the raw LLM response classified it as `theme:001_pricing`.
- The route became `manager_only` only after the safety layer detected a high-risk marker outside the theme itself.
- This was too fragile because a slightly different LLM response could miss the marker.

Implemented fix:

- Added input-level high-risk detection before trusting LLM theme classification.
- Refund, matkap, tax, legal, complaint, discount/installment and payment-status wording in the client message now force `manager_only`.
- The guard works even when LLM returns the wrong `topic_id`.

No live systems were touched.
