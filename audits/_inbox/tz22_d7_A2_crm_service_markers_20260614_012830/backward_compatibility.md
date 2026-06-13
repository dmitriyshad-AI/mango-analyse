# Backward Compatibility

## Compatibility Status

Compatible for clean CRM payloads.

## Existing Behavior Preserved

- Clean customer text still returns no findings.
- Manual `История общения` remains outside the new service marker gate.
- Existing severity filtering already supports `P0`, so `min_severity="P2"` includes the new risk.
- Existing detector API signatures are unchanged.

## Intentional Behavior Change

- Generated CRM text containing explicit service/test markers is now blocked as `P0`.

## Data Checks Not Run

- The 50-row snapshot check was not run because ignored product snapshots are not present in this worktree.
