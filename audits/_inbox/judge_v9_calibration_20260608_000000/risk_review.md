# Risk review

Primary risk: judge v9 could change measurement labels if accidentally enabled during an ongoing v2 measurement series.

Controls:

- CLI default is `--judge-prompt-version v2`.
- Summary records both normalized version and stable prompt id.
- `CLAUDE.md` now forbids comparing runs with different judge prompt versions without re-judging both sides.

Runtime risk: broad fact-claim extraction could affect output gates if shared code changed behavior.

Controls:

- New extraction is behind `include_judge_generic_claims=False` by default.
- Existing runtime callers retain default behavior.
- Existing fact-audit tests still pass.

Measurement risk: judge re-ask could silently change verdicts.

Controls:

- Re-ask runs only for v9 and completed FAIL dialogs with missing/unspecified hard gates.
- Re-ask only updates concrete `violated_gates`.
- A PASS verdict from re-ask is ignored.

Residual risk:

- v9 prompt calibration still needs live/regression re-judging on saved runs before it becomes a trusted comparison baseline.
- The sidecar re-judge script intentionally depends on saved per-turn transcript fields; incomplete older transcripts may produce weaker judge context.
