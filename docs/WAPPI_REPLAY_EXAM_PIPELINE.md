# Wappi Replay Exam Pipeline

## Scope

`chat_only_replay` measures how the bot draft compares with a manager reply on scrubbed Wappi chat turns. It is not a full live-parity metric: external CRM/Tallanto checks, manager private work and non-chat context are segmented separately.

## Safety Contract

- Raw export is allowed only under `~/.mango_local/replay_exam/raw/`.
- Wappi reads must use paginated `get_chat_messages(..., mark_all=False)`.
- `AmoWappiDraftLoop.run_once`, AMO note clients, AiOffice note clients, state, journal and heartbeat are not part of the exporter.
- Live Wappi read requires separate owner confirmation and the `--allow-live-wappi-read` CLI flag.
- Scrubbed sets, not raw dumps, are used by runner/judge/M1.

## Pipeline

1. Export raw Wappi messages read-only.
2. Pseudonymize every field recursively: message body, names, manager reference, traces and judge payloads.
3. Slice dialogs with teacher forcing: all client turns are replayed in chronology; only turns with a manager reference are scored.
4. Run bot drafts offline, in parallel by dialog only.
5. Run machine gate first: unsupported numbers, wrong brand, PII, P0 route/flags.
6. Run `replay_judge_v1` only after machine gate, with hidden A/B mapping.
7. Build M1 manifest on the final accepted HEAD.

## Current Status

Implemented and unit-tested offline foundation:

- `mango_mvp.replay_exam.exporter`
- `mango_mvp.replay_exam.pseudonymizer`
- `mango_mvp.replay_exam.slicer`
- `mango_mvp.replay_exam.runner`
- `mango_mvp.replay_exam.machine_gate`
- `mango_mvp.replay_exam.judge`
- `mango_mvp.replay_exam.m1_adapter`

Not run yet: live Wappi export and real pilot-10, because live-read needs explicit owner confirmation.
