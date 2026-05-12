# Local Throughput Profile: M4 Max

Purpose: run Mango analyse passes faster without trading away data safety.

## Main rule

Different stages bottleneck on different resources:

- ASR uses local hardware. It can and should load the Mac heavily.
- Resolve and Analyze mostly wait on Codex/LLM calls and SQLite coordination. They are accelerated by multiple worker processes, not by one larger process.
- Deterministic reports/backfills are usually fast. Increase batch size, but do not overcomplicate them.

## ASR-only profile

Use this when the goal is maximum local transcription throughput.

Recommended settings:

```zsh
export MANGO_UI_TRANSCRIBE_MODE="dual"
export MANGO_UI_TRANSCRIBE_PROVIDER="mlx"
export MANGO_UI_SECONDARY_PROVIDER="gigaam"
export MANGO_UI_MERGE_PROVIDER="rule"
export MANGO_UI_PIPELINE_STAGE_TRANSCRIBE="1"
export MANGO_UI_PIPELINE_STAGE_BACKFILL="1"
export MANGO_UI_PIPELINE_STAGE_RESOLVE="0"
export MANGO_UI_PIPELINE_STAGE_ANALYZE="0"
export OMP_NUM_THREADS="6"
export MKL_NUM_THREADS="6"
export KMP_DUPLICATE_LIB_OK="TRUE"
export KMP_INIT_AT_FORK="FALSE"
```

Practical notes:

- Keep one GigaAM process with up to 6 CPU threads. Multiple GigaAM processes can fight for memory/CPU and slow the machine down.
- Whisper MLX and GigaAM together are the useful dual-ASR setup: Whisper loads Apple acceleration, GigaAM uses CPU.
- If the Mac is also doing Resolve/Analyze, ASR speed will drop. For maximum ASR speed, run ASR-only.

## Resolve + Analyze profile

Use this when ASR is already done and the goal is to finish R+A quickly.

Default M4 Max launcher:

```zsh
stable_runtime/start-ra-max-detached.sh \
  --db "stable_runtime/<batch>/<batch>.db" \
  --resolve-workers 2 \
  --analyze-workers 6 \
  --stage-limit 20 \
  --poll-sec 5
```

Monitor:

```zsh
tail -f "stable_runtime/runs/<run_id>/controller.log"
```

Why 2 + 6:

- Resolve is usually cheaper and can feed Analyze fast enough with 2 workers.
- Analyze is the expensive LLM stage, so 6 workers gives higher throughput.
- More than 6 Analyze workers may help sometimes, but it also increases Codex rate-limit risk, retries, and SQLite lock contention.

Safety settings used by the launcher:

```zsh
export SQLITE_WAL_ENABLED="1"
export SQLITE_BUSY_TIMEOUT_MS="60000"
export LLM_CACHE_ENABLED="1"
export RESOLVE_MAX_ATTEMPTS="20"
export ANALYZE_MAX_ATTEMPTS="20"
export CODEX_CLI_TIMEOUT_SEC="240"
```

## Analyze-only profile

Use this if Resolve is already fully done/skipped:

```zsh
stable_runtime/start_analyze_refresh_detached.sh \
  --db "stable_runtime/<batch>/<batch>.db" \
  --concurrency 6 \
  --batch-limit 20 \
  --skip-reset
```

Do not use `--skip-reset` if you intentionally want to reanalyze old done rows.

## Deterministic audit/backfill profile

For guardrails, reports, and backfills:

```zsh
PYTHONPATH=src python3 scripts/build_transcript_quality_guardrails_dry_run.py \
  --database-url sqlite:///stable_runtime/<batch>/<batch>.db \
  --out-root stable_runtime/<out> \
  --batch-size 2000
```

These stages usually do not need worker fan-out. Keep them single-process to avoid unnecessary DB complexity.

## When to increase workers

Increase Analyze workers from 6 to 8 only if all are healthy:

- controller log shows steady progress;
- no repeated Codex timeouts;
- no frequent SQLite busy/locked errors;
- CPU and memory are not pressuring ASR or other active work.

Decrease workers if:

- many `failed`/`dead_letter` rows appear;
- Codex calls time out;
- the Mac starts swapping;
- ASR is running and must keep priority.
