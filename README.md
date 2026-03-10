# Mango Calls -> amoCRM MVP

Working MVP scaffold to process exported Mango call recordings in batch:

1. ingest audio files
2. transcribe calls
3. extract sales insights
4. sync notes/custom fields/tasks to amoCRM contacts

This repo is intentionally simple and safe for first launch:
- SQLite by default
- dry-run sync enabled by default
- `mock` providers available for smoke tests without API costs

## Requirements

- Python 3.9+
- `ffprobe` and `ffmpeg` in PATH (recommended for metadata and channel split)
  - if `ffprobe` is missing on macOS, ingest now falls back to `afinfo` for duration/rate/channels
- OpenAI API key (if using `openai` providers)
- amoCRM OAuth credentials (if disabling `SYNC_DRY_RUN`)

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env
```

Initialize DB:

```bash
mango-mvp init-db
```

If `mango-mvp` is not in PATH, use:

```bash
python3 -m mango_mvp.cli <command>
```

## Git + CI/CD (auto deploy)

The project includes Git/bootstrap and deployment automation:

- bootstrap remote and first push: `scripts/git_bootstrap.sh`
- optional auto commit/push loop: `scripts/start_autocommit_push.sh` and `scripts/stop_autocommit_push.sh`
- CI workflow: `.github/workflows/ci.yml`
- auto deploy workflow: `.github/workflows/deploy-server.yml`
- server rebuild script: `deploy/server_rebuild.sh`

Setup guide:

- `docs/GIT_CICD_AUTODEPLOY.md`

## Run pipeline

### Step-by-step

```bash
mango-mvp ingest --recordings-dir /absolute/path/to/mango_export --metadata-csv /absolute/path/to/calls.csv
mango-mvp transcribe --limit 100
mango-mvp resolve --limit 100
mango-mvp analyze --limit 100
mango-mvp sync --limit 100
mango-mvp worker --stage-limit 100
mango-mvp requeue-dead --stage all --limit 1000
mango-mvp reset-transcribe --only-missing-variants --only-done --limit 1000
mango-mvp export-review-queue --out /absolute/path/manual_review_queue.csv --limit 10000
mango-mvp export-failed-resolve-queue --out /absolute/path/failed_resolve_queue.csv --limit 10000
mango-mvp export-crm-fields --out /absolute/path/crm_fields.csv --only-done --limit 100000
mango-mvp migrate-analysis-schema --only-done --limit 10000 --dry-run
mango-mvp stats
```

### One command

```bash
mango-mvp run-all --recordings-dir /absolute/path/to/mango_export --metadata-csv /absolute/path/to/calls.csv --stage-limit 100
```

Resilient background worker:

```bash
mango-mvp worker --stage-limit 100 --poll-sec 10 --max-idle-cycles 0
```

Set `--max-idle-cycles 0` for infinite loop mode.

## Metadata CSV (optional)

If you pass `--metadata-csv`, the ingest step maps rows by recording filename.
Supported column aliases:

- filename: `filename`, `file_name`, `recording`, `record`, `audio_file`
- phone: `phone`, `client_phone`, `abonent_number`, `contact_phone`
- call id: `call_id`, `id`, `record_id`
- manager: `manager`, `manager_name`, `operator`
- direction: `direction`, `call_direction`
- started at: `started_at`, `start_time`, `date_time`

Template file: `examples/metadata_template.csv`

If CSV is missing/incomplete, ingest now also parses fallback metadata directly
from Mango filename patterns:

- `YYYY-MM-DD__HH-MM-SS__Менеджер__PHONE_CALLID.mp3`
- `YYYY-MM-DD__HH-MM-SS__PHONE__Менеджер_CALLID.mp3`

Fallback fields:
- `phone`
- `manager_name`
- `source_call_id`
- `started_at`

## Providers

Environment flags:

- `TRANSCRIBE_PROVIDER=mock|openai|mlx`
- `ANALYZE_PROVIDER=mock|openai|ollama|codex_cli`
- `TRANSCRIPT_EXPORT_DIR=transcripts` (set empty value to disable .txt export)
- `DUAL_TRANSCRIBE_ENABLED=true|false`
- `SECONDARY_TRANSCRIBE_PROVIDER=gigaam|mlx|openai|mock`
- `DUAL_MERGE_PROVIDER=primary|rule|openai|ollama|codex_cli`
- `OPENAI_MERGE_MODEL=gpt-4o-mini`
- `CODEX_MERGE_MODEL=gpt-5.4`
- `CODEX_CLI_COMMAND=codex`
- `CODEX_CLI_TIMEOUT_SEC=120`
- `DUAL_MERGE_SIMILARITY_THRESHOLD=0.985`
- `MONO_ROLE_ASSIGNMENT_MODE=off|rule|openai_selective|ollama_selective`
- `MONO_ROLE_ASSIGNMENT_MIN_CONFIDENCE=0.62`
- `MONO_ROLE_ASSIGNMENT_LLM_THRESHOLD=0.72`
- `OPENAI_ROLE_ASSIGN_MODEL=gpt-4o-mini`
- `OLLAMA_BASE_URL=http://127.0.0.1:11434`
- `OLLAMA_MODEL=gpt-oss:20b`
- `OLLAMA_THINK=medium`
- `OLLAMA_TEMPERATURE=0`
- `ANALYZE_OLLAMA_NUM_PREDICT=500`
- `TRANSCRIBE_MAX_ATTEMPTS=3`
- `RESOLVE_MAX_ATTEMPTS=2`
- `ANALYZE_MAX_ATTEMPTS=3`
- `SYNC_MAX_ATTEMPTS=3`
- `RESOLVE_MIN_DURATION_SEC=30`
- `RESOLVE_LLM_TRIGGER_SCORE=75`
- `RESOLVE_ACCEPT_SCORE=75`
- `RESOLVE_LLM_PROVIDER=off|ollama|openai|codex_cli`
- `RESOLVE_LLM_FOR_RISKY=false` (enable LLM-resolve for risky turn-order/timing calls)
- `RESOLVE_RESCUE_PROVIDER=` (optional, auto if empty)
- `RESOLVE_RESCUE_DUAL_ENABLED=false`
- `RESOLVE_POSTFILTER_SAME_TS=true`
- `RESOLVE_RISKY_SAME_TS_THRESHOLD=2`
- `RESOLVE_AGGRESSIVE_RESCUE_FOR_RISKY=true`
- `RETRY_BASE_DELAY_SEC=30`
- `WORKER_POLL_SEC=10`
- `WORKER_MAX_IDLE_CYCLES=30`

`.env.example` is prefilled with a recommended Codex-based pipeline for this project
(`mlx + gigaam + codex_cli`). For fully local no-cloud runs, switch merge/analyze/resolve
providers to `rule`/`mock`/`off` or use the local semi-prod profile.

With `openai`:
- transcription model is `OPENAI_TRANSCRIBE_MODEL` (`gpt-4o-transcribe` by default)
- merge model is `OPENAI_MERGE_MODEL` (`gpt-4o-mini` by default)
- analysis model is `OPENAI_ANALYSIS_MODEL` (`gpt-4o-mini` by default)
- resolve LLM (for risky calls) uses `RESOLVE_LLM_PROVIDER=openai` and `OPENAI_MERGE_MODEL`

With `codex_cli` (cloud merge via Codex login):
- run `codex login` once and ensure `codex login status` says "Logged in"
- dual merge uses `DUAL_MERGE_PROVIDER=codex_cli`
- resolve merge uses `RESOLVE_LLM_PROVIDER=codex_cli`
- analyze uses `ANALYZE_PROVIDER=codex_cli`
- model is `CODEX_MERGE_MODEL` (default `gpt-5.4`)

With `ollama` (local LLM for analyze/merge/mono-role):
- default base URL is `OLLAMA_BASE_URL` (`http://127.0.0.1:11434`)
- model is `OLLAMA_MODEL` (`gpt-oss:20b` by default)
- thinking level is `OLLAMA_THINK` (`medium` by default)
- deterministic output: keep `OLLAMA_TEMPERATURE=0`

With `mlx` (local Apple Silicon inference):
- install packages: `python3 -m pip install -r requirements-local-whisper.txt`
- create local ffmpeg shim for `mlx-whisper`:
  - `mkdir -p .local/bin`
  - `ln -sf "$(python3 -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')" .local/bin/ffmpeg`
  - run CLI commands with `PATH="$(pwd)/.local/bin:$PATH"`
- model repo is `MLX_WHISPER_MODEL` (`mlx-community/whisper-large-v3-mlx` by default)
- repetition-loop guard for long calls: set `MLX_CONDITION_ON_PREVIOUS_TEXT=false`
- better turn ordering/timing: keep `MLX_WORD_TIMESTAMPS=true` (default)

With `gigaam` (optional second-pass merge):
- requires Python `>=3.10` in the environment where pipeline runs
- install package: `python3 -m pip install -r requirements-local-dual-asr.txt`
- set `SECONDARY_TRANSCRIBE_PROVIDER=gigaam`
- model and chunking params: `GIGAAM_MODEL`, `GIGAAM_DEVICE`, `GIGAAM_SEGMENT_SEC`

## Dual ASR Merge

When enabled, pipeline transcribes each channel twice:
1. primary provider (`TRANSCRIBE_PROVIDER`)
2. secondary provider (`SECONDARY_TRANSCRIBE_PROVIDER`)

Then final text is built by `DUAL_MERGE_PROVIDER`:

- `primary`: keep only variant A
- `rule`: deterministic merge (token alignment + anti-loop heuristics)
- `openai`: LLM merge from 2 independent variants (fallback to `rule` on API errors)
- `ollama`: local LLM merge from 2 independent variants (fallback to `rule` on errors)
- `codex_cli`: cloud merge through `codex exec` (ChatGPT login), fallback to `rule` on errors

Optimization:
- if A/B are very similar (`DUAL_MERGE_SIMILARITY_THRESHOLD`), pipeline keeps A without LLM call.

For `codex_cli` merge:
- run `codex login` once and ensure `codex login status` says "Logged in";
- set `DUAL_MERGE_PROVIDER=codex_cli`;
- optional model override via `CODEX_MERGE_MODEL` (default `gpt-5.4`).
- if selected model is unavailable, pipeline falls back to `rule`.

## Stereo handling

If `SPLIT_STEREO_CHANNELS=true` and recording has 2 channels:
- left channel is treated as manager
- right channel is treated as client

If split fails, pipeline falls back to full-file transcription.

## Mono Role Assignment (Selective LLM)

For `mono_or_fallback` calls (channels are not reliably separable):

- `off`: keep timeline as `Спикер (не определен)`
- `rule`: local rule-based split into `Менеджер/Клиент`
- `openai_selective`: first try local rules, and call LLM only on low-confidence cases
- `ollama_selective`: first try local rules, and call local Ollama only on low-confidence cases

Selective mode reduces LLM usage: confident calls stay local, only uncertain calls
go to LLM (`OPENAI_ROLE_ASSIGN_MODEL` for `openai_selective`, `OLLAMA_MODEL` for `ollama_selective`).

## Transcript files export

After successful transcribe, pipeline writes a `.txt` copy for each call to:

- `<TRANSCRIPT_EXPORT_DIR>/<parent_audio_folder>/<audio_stem>_text.txt`
- `<TRANSCRIPT_EXPORT_DIR>/<parent_audio_folder>/<audio_stem>_variants.json`
- `<TRANSCRIPT_EXPORT_DIR>/<parent_audio_folder>/<audio_stem>_history_summary.txt`
- `<TRANSCRIPT_EXPORT_DIR>/<parent_audio_folder>/<audio_stem>_structured_fields.json`

Format:

- `[MM:SS.s] Менеджер (<ФИО из имени файла>): ...`
- `[MM:SS.s] Клиент: ...`
- `[~MM:SS] ...` means estimated time (no exact segment timestamps from ASR)

Phrases are written in chronological dialog order. For mono/fallback mode
(`каналы не разделены`) text is still split into a sequential timeline with
`Спикер (не определен)`.

If left/right channels look mirrored (very high text similarity),
pipeline auto-falls back from stereo roles to mono timeline to avoid
false manager/client split. Tune via:
- `STEREO_OVERLAP_SIMILARITY_THRESHOLD` (default `0.97`)
- `STEREO_OVERLAP_MIN_CHARS` (default `80`)

`_variants.json` contains both independent ASR hypotheses (`variant_a`, `variant_b`),
selected final text, and merge metadata (`selection`, `provider`, `confidence`, `notes`).

## Resolve stage (LLM + Rescue ASR)

After transcription you can run:

```bash
mango-mvp resolve --limit 100
```

What it does:
- skips short calls `< RESOLVE_MIN_DURATION_SEC` (default: 30s);
- calculates quality score for current transcript;
- for risky calls runs contextual LLM merge (`RESOLVE_LLM_PROVIDER=ollama|openai|codex_cli`);
- recalculates quality;
- if still weak, runs rescue re-transcribe profile with alternate ASR provider;
- chooses best result automatically;
- sends unresolved low-quality calls to `manual` queue.

Manual queue export:

```bash
mango-mvp export-review-queue --out manual_review_queue.csv --limit 10000
mango-mvp export-failed-resolve-queue --out failed_resolve_queue.csv --limit 10000
mango-mvp export-crm-fields --out crm_fields.csv --only-done --limit 100000
```

## Operational reliability

Current reliability controls:
- automatic retries with exponential backoff
- terminal status `dead` after max attempts
- `dead_letter_stage` for failure stage tracking
- `next_retry_at` scheduling field
- independent retry limits per stage: transcribe/resolve/analyze/sync
- `worker` command for unattended background processing
- `requeue-dead` command for replay from dead-letter
- `reset-transcribe` command to re-run already processed calls (for example to fill missing variants)

## Analysis guardrails

Analyze stage auto-detects non-conversation calls (voicemail/assistant/callback stubs)
and stores a structured `non_conversation` result instead of forcing sales analysis.

For real conversations, expected structured blocks include:
- summary
- interests
- student_grade
- target_product
- personal_offer
- pain_points
- budget
- timeline
- objections
- next_step
- follow_up_score / follow_up_reason

Analysis payloads are versioned (`analysis_schema_version`).  
Current schema version is `v2` (includes `history_short`, `crm_blocks`, `evidence`, `quality_flags`
plus new aliases `history_summary`, `structured_fields` and legacy-compatible keys).

Upgrade already analyzed calls without re-transcribe:

```bash
mango-mvp migrate-analysis-schema --only-done --limit 10000 --dry-run
mango-mvp migrate-analysis-schema --only-done --limit 10000
```

## amoCRM sync

Current sync logic:

- find contact by phone (`query` with last 10 digits)
- add AI summary as contact note
- update mapped custom fields if IDs configured
- create follow-up task when score >= `FOLLOW_UP_TASK_THRESHOLD`

Useful custom-field mappings:
- `AMOCRM_INTERESTS_FIELD_ID`
- `AMOCRM_STUDENT_GRADE_FIELD_ID`
- `AMOCRM_TARGET_PRODUCT_FIELD_ID`
- `AMOCRM_PERSONAL_OFFER_FIELD_ID`

Run safe test first:

```bash
SYNC_DRY_RUN=true
```

Then switch to production:

```bash
SYNC_DRY_RUN=false
```

## Token budget estimation

Use script to estimate LLM tokens for:
- dual-ASR merge (A/B -> final)
- structured analysis JSON

Example:

```bash
python3 scripts/estimate_token_budget.py --database pilot_mlx.db --calls 100000 --merge-rate 0.35
```

## Visual UI

Desktop UI (Tkinter, no extra dependencies):

```bash
mango-mvp-ui
```

If script is unavailable in PATH:

```bash
PYTHONPATH=src python3 -m mango_mvp.gui
```

If your active Python/venv has no Tk support (`ModuleNotFoundError: _tkinter`), launch UI
with macOS system Python and keep ASR execution in your venv via `Backend Python` field:

```bash
cd "/absolute/path/to/project"
PYTHONPATH=src PATH="$PWD/.local/bin:$PATH" /usr/bin/python3 -m mango_mvp.gui
```

From UI you can:
- pick recordings folder and metadata CSV
- pick DB file and transcript output folder
- pick backend Python and toggle `Use project src (dev mode)`
- configure transcribe/merge/analyze modes
- run stage buttons (`init-db`, `ingest`, `transcribe`, `analyze`, `sync`, `stats`)
- start/stop background worker

## Stable standalone runtime snapshot

To run a stable isolated build (separate from ongoing source edits), use:

```bash
cd "/absolute/path/to/project"
./stable_runtime/run-ui.sh
```

Stable CLI launcher:

```bash
./stable_runtime/run-cli.sh --help
```

Snapshot metadata and locked dependencies:

- `stable_runtime/SNAPSHOT_CREATED_AT.txt`
- `stable_runtime/requirements-lock.txt`

## Semi-production overnight run (300-500 calls)

One-command detached launcher (start and leave it running in background):

```bash
cd "/absolute/path/to/project"
./stable_runtime/start_semi_prod_detached.sh \
  --input "/absolute/path/to/mango_export" \
  --output "/absolute/path/to/transcripts_semi_prod" \
  --target 400 \
  --stage-limit 30 \
  --qc-every 50
```

What it does:
- creates/uses dedicated DB (`stable_runtime/semi_prod.db` by default)
- ingests up to `--target` calls
- runs transcribe loop in batches
- auto-builds QC reports every `--qc-every` calls and final QC at the end

Artifacts per run:
- `stable_runtime/runs/<run_id>/run.log`
- `stable_runtime/runs/<run_id>/qc_*.json`
- `stable_runtime/runs/<run_id>/final_stats.json`

Default semi-prod profile (local deterministic, no cloud LLM):
- `stable_runtime/profiles/semi_prod_dual_asr.env`

Codex profile (dual ASR + Codex merge/resolve/analyze):
- `stable_runtime/profiles/semi_prod_dual_asr_codex_merge.env`
- detached launcher: `./stable_runtime/start_codex_merge_detached.sh ...`

OpenAI profile:
- `stable_runtime/profiles/semi_prod_dual_asr_openai_merge.env`

Targeted runner to reach a fixed analyzed volume (default `1000`) with progress
checkpoints every 20 calls:

```bash
./stable_runtime/fill_to_1000.sh
```

## Smoke tests

Run basic operational checks:

```bash
make test-smoke
```

## Suggested next improvements

1. Add adaptive load profiles (CPU/RAM-aware throttling for long batch runs)
2. Add scheduled quality report + auto reprocess queue for risky transcripts
3. Add staged amoCRM rollout with field validation and sync error report
4. Add dashboard (conversion by manager, objections, top interests)
5. Add release/versioning workflow (tagged stable snapshots + rollback points)
