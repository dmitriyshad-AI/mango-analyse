# Risk review

Risk level: medium-low.

Checked risks:
- Identity self-disclosure: output safety-net is now present in v2.
- False identity match by substring: boundary matching added and tested.
- Precedence conflict: cross-brand remains higher priority than terminal.
- Legacy path: unchanged except safer phrase detector.

Known uncovered risk:
- Some terminal direct-info outputs contain concrete phone/address values. If v2 retrieval did not include the matching fact, re-verification will fail closed to safe fallback. This is safer than leaking facts, but may be less useful until fact retrieval is validated on live runs.

