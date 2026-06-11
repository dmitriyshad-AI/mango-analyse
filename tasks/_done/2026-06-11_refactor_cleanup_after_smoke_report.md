# Refactor cleanup after smoke PASS — 2026-06-11

## Context

Architect confirmed refactor smoke: PASS, hard gates zero, critical classes clean by raw transcripts.

## Actions

Moved untracked run artifacts from old worktrees into the main folder:

- `Mango analyse.tz13-main/runs/20260611_memory_provenance_smoke18_on` -> `runs/_from_worktrees/tz13-main/`
- `Mango analyse.tz13-main/runs/20260611_memory_provenance_smoke18_on_retry` -> `runs/_from_worktrees/tz13-main/`
- `Mango analyse.wave1-probe/runs/20260608_wave1_probe_gpt_OFF` -> `runs/_from_worktrees/wave1-probe/`
- `Mango analyse.wave1-probe/runs/20260608_wave1_probe_gpt_OFF_isolated` -> `runs/_from_worktrees/wave1-probe/`
- `Mango analyse.wave1-probe/runs/20260608_wave1_probe_gpt_ON_isolated` -> `runs/_from_worktrees/wave1-probe/`

Removed worktrees:

- `/Users/dmitrijfabarisov/Projects/Mango analyse.tz13-main`
- `/Users/dmitrijfabarisov/Projects/Mango analyse.wave1-probe`
- `/Users/dmitrijfabarisov/Projects/Mango analyse.refactor`

Deleted merged branch:

- `codex/refactor-subscription-llm` at `8402b776`

Ran:

```bash
git worktree prune
```

## Final state

`git worktree list --porcelain` now shows only the main project folder:

- `/Users/dmitrijfabarisov/Projects/Mango analyse`
- branch `main`
- head `c1101e91`

Tracked tree status:

```text
## main...origin/main [ahead 425]
```

Full `git status` still shows pre-existing untracked project artifacts in the main folder (`runs/`, `_bundles/`, `D1_audit_backlog/*`, `tasks/_inbox_codex/*`, local scripts, old KB copies, etc.). They were outside the explicit cleanup scope and were not deleted or hidden.

