# Semantic Review

Status: semantic pass for report-only diagnostics.

This report tests the auditor's objection: changing only `requested_action` is not enough when the frame still says `risk_class=manager_action`, `answerability=manager_only`, `must_handoff=true`.

On the real F2y current handoff cases:

- 1 scope-confusion case exists.
- action-only counterfactual is still blocked in 4/5 cases.
- full safe-reference counterfactual finds exact product proof in 3/5 cases.
- 4/5 cases still have residual hard missing axes.
- 0 new active candidates.

Business conclusion:

- Do not enable F3.
- Do not patch only `requested_action`.
- Calibrate the whole frame tuple if this class is pursued: `requested_action`, `risk_class`, `answerability`, `must_handoff`.
- Even after proof recovery, residual hard axes still need separate policy/fact work.
