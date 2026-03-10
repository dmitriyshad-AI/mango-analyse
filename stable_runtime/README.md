# Stable Runtime

This folder contains a snapshot runtime that is isolated from day-to-day source edits.

## What is isolated

- Python environment: `stable_runtime/venv_stable`
- Installed app package snapshot (`mango-call-mvp`) in that venv
- Launchers that do not depend on `PYTHONPATH=src`

## Run

```bash
cd "/Users/dmitrijfabarisov/Projects/Mango analyse"
./stable_runtime/run-ui.sh
```

CLI from stable runtime:

```bash
./stable_runtime/run-cli.sh --help
./stable_runtime/run-cli.sh stats
./stable_runtime/run-cli.sh resolve --limit 100
./stable_runtime/run-cli.sh export-review-queue --out /tmp/manual_review_queue.csv
./stable_runtime/run-cli.sh export-failed-resolve-queue --out /tmp/failed_resolve_queue.csv
./stable_runtime/run-cli.sh export-crm-fields --out /tmp/crm_fields.csv --only-done --limit 100000
./stable_runtime/run-cli.sh migrate-analysis-schema --only-done --limit 10000 --dry-run
```

Standalone converter with optional mono-role assignment:

```bash
./stable_runtime/audio2text.sh --help
# Example:
./stable_runtime/audio2text.sh \
  --input /path/to/calls \
  --output /path/to/transcripts \
  --provider mlx \
  --mlx-word-ts 1 \
  --mono-assign openai_selective
```

Use Codex CLI merge (ChatGPT login, no project API key):

```bash
codex login
./stable_runtime/audio2text.sh \
  --input /path/to/calls \
  --output /path/to/transcripts_codex_merge \
  --provider mlx \
  --dual \
  --secondary gigaam \
  --merge codex_cli
```

Semi-production overnight run (detached, one command):

```bash
./stable_runtime/start_semi_prod_detached.sh \
  --input /path/to/calls \
  --output /path/to/transcripts_semi_prod \
  --target 400 \
  --stage-limit 30 \
  --qc-every 50
```

Cloud merge profile (OpenAI merge instead of local `gpt-oss`):

```bash
OPENAI_API_KEY=... \
./stable_runtime/start_openai_merge_detached.sh \
  --input /path/to/calls \
  --output /path/to/transcripts_openai_merge \
  --target 300 \
  --stage-limit 20 \
  --qc-every 50
```

Cloud merge profile via Codex CLI login:

```bash
codex login
CODEX_MERGE_MODEL=gpt-5.4 \
./stable_runtime/start_codex_merge_detached.sh \
  --input /path/to/calls \
  --output /path/to/transcripts_codex_merge \
  --target 300 \
  --stage-limit 20 \
  --qc-every 50
```

Parallel fill to 1000 (faster end-to-end on one Mac: ASR + Codex stages in parallel):

```bash
./stable_runtime/fill_to_1000_parallel.sh \
  --db "/Users/dmitrijfabarisov/Projects/Mango analyse/stable_runtime/semi_prod_400.db" \
  --target-done 1000 \
  --transcribe-limit 6 \
  --resolve-limit 10 \
  --analyze-limit 10 \
  --progress-every 20
```

Detached mode for overnight run:

```bash
./stable_runtime/start_fill_to_1000_parallel_detached.sh

# Optional explicit params:
./stable_runtime/start_fill_to_1000_parallel_detached.sh \
  --db "/Users/dmitrijfabarisov/Projects/Mango analyse/stable_runtime/semi_prod_400.db" \
  --target-done 1000
```

Per-run artifacts:
- `stable_runtime/runs/<run_id>/run.log`
- `stable_runtime/runs/<run_id>/qc_*.json`
- `stable_runtime/runs/<run_id>/final_stats.json`
- `stable_runtime/runs/<run_id>/run_config.txt`

Default semi-prod profile:
- `stable_runtime/profiles/semi_prod_dual_asr.env`
- edit it once if you want different defaults for all future overnight runs

## Notes

- Future edits in `src/` do not affect this runtime until you rebuild or reinstall.
- Audio/model files are still shared from the same machine and paths.
- `rebuild_snapshot.sh` uses `requirements-lock.txt` if present, then reinstalls local project wheel
  without dependency upgrades.
