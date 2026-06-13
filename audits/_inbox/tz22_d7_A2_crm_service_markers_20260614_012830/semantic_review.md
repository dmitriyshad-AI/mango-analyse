# Semantic Review

Verdict: PASS_WITH_NOTES

## Artifact And Audience

- Artifact: CRM text quality gate for AMO/contact writeback text.
- Audience: CRM/AMO managers and internal operators.

## What Passed

- Service/test markers in auto/AI CRM fields now produce blocking `P0` risk `service_test_marker`.
- The detector blocks the confirmed corruption class: `smoke test`, `AI Office`, `Тестовый ИИ`, `match-status`, `ai-priority`.
- The detector avoids broad `тест` matching, so normal phrases like `дз по мат в виде тестов` are not blocked.
- The detector is scoped to generated CRM text fields, not manual `История общения`.

## Blocking Issues

- None in code/fixtures.

## Non-Blocking Risks

- Raw 50-row snapshot check was not run in this worktree because ignored product snapshots are absent. Architect must run this on source data.
- The live corrupted contact 76062310 is not cleaned by this code change; data cleanup remains a manual AMO action.

## Missing Checks

- Source-data regрейд on `product_data/customer_profiles/tz14_amo_step1_full_20260612/amo_contacts_raw.jsonl`.

## Regression Rule

Confirmed semantic bug: service/test text leaked into `Авто история общения`.

Permanent checks added:

- `test_blocks_service_test_marker_in_auto_history`
- `test_service_test_marker_does_not_match_plain_test_word`
- `test_service_test_marker_ignores_manual_history_field`
- `test_service_test_marker_ignores_manual_history_even_with_marker`
- `test_service_test_marker_does_not_match_test_history_in_auto_field`
