# Risk Review

## Changed Behavior

- Generated CRM text fields with explicit service/test markers are now blocked as P0.
- Manual field `История общения` is not blocked by the new detector.

## Safety Risks

- False positives: controlled by a narrow regex and strict field scope.
- False negatives: possible for new service marker variants not listed in the regex; this is acceptable because the detector targets the confirmed corruption class.
- Write risk: no live write paths were executed.

## Adversarial / Edge Cases Checked

- `smoke test` / `AI Office` / `match-status` / `ai-priority` / `Тестовый ИИ` in `Авто история общения` -> blocked.
- `дз по мат в виде тестов` -> not blocked.
- `Тестовая история` in manual field -> not blocked.
- `AI Office smoke test` in manual field -> not blocked.
- `Тестовая история` in auto field -> not blocked.
