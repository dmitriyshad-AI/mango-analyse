# Risk Review

Primary risk if misused: treating counterfactual proof as permission to change runtime behavior.

Mitigations:

- Report-only script.
- `active_behavior_allowed=false` for every case.
- `new_active_candidates=0`.
- No route/text/profile/live changes.
- Negative controls remain preserved.
- Residual hard missing axes are explicit.

Residual risks:

- Counterfactual proof is an offline diagnostic, not production evidence.
- It does not generate or validate client-facing wording.
- It does not solve live availability, payment access, price precision, dates or location coverage.
