# TZ-19 calls review table risk review

## Main risks

- PII leakage to git.
- Accidental live writes or reruns.
- Confusing the 22,679 baseline v7 scope with current post-TZ21 26,118 v7 rows.

## Controls

- Workbook path is outside repository: `/Users/dmitrijfabarisov/Claude Projects/Foton/`.
- Only script, tests, report, and audit pack are tracked.
- Canonical DB opened read-only.
- Script does not import AMO/Tallanto clients and has no write path.
- ASR/Analyse/R+A are not called.
- Default scope excludes TZ-21 tail ids to match the 22,679-row ТЗ.

## Residual risk

The workbook remains sensitive even after masking because call summaries can contain business context and possible indirect identifiers. It must stay outside git and should not be sent to external tools without a separate decision.
