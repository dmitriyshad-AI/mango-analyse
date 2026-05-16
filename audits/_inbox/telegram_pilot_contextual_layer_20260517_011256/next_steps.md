# Next Steps

1. Wait for Claude's end-to-end testing review.
2. Review the Stage 6 smoke CSV manually:
   - `.codex_local/telegram_pilot/eval_packs/20260517_contextual_layer_smoke/private_manual_review.csv`
   - check whether it contains enough family/multiple-children, prompt-injection, refund, payment, schedule, matkap, non-question and wait-for-more examples.
3. If the smoke sample is too easy, build a second targeted pack with hard cases.
4. Run LLM draft generation only on the reviewed 20-30 historical dialogs.
5. Review drafts manually with Dmitry.
6. If safe, run limited live with Dmitry test accounts and Nastya service chat.
7. Only after limited live, decide whether to open one controlled website entry point.
8. Start 100-300 contextual question sample before any 9 969 full contextual run.
