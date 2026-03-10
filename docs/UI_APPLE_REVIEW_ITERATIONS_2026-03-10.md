# UI Iterations (Apple-style review simulation)

This review is a **simulated Apple HIG-style persona**, not an actual Apple employee.

## Iteration 1 - hierarchy and focus

Reviewer comment:
- "The UI mixes operational and technical controls in one long screen. Primary actions should be obvious in 3 seconds."

Changes applied:
- moved to tabbed layout: `Pipeline`, `Status`, `Functions`, `Design Review`
- added clear action groups with primary buttons
- added explicit run-state indicator and stop button

## Iteration 2 - operational reliability

Reviewer comment:
- "Long-running data tools must show queue status and progress at a glance."

Changes applied:
- added status cards (done/pending/failed/manual/dead)
- added transcribe/analyze completion progress bars
- added periodic stats refresh and raw JSON status panel

## Iteration 3 - workflow separation

Reviewer comment:
- "Transcription and post-processing are different operator intents. Keep them independently executable."

Changes applied:
- independent transcription mode switch: `whisper`, `gigaam`, `dual`
- dedicated Codex post-processing actions:
  - `Codex Resolve Batch`
  - `Codex Analyze Batch`
  - `Codex Post-process All`
- explicit idempotency note (done rows are not reprocessed by default)

## Iteration 4 - operator onboarding

Reviewer comment:
- "Every action should be self-documented inside the UI."

Changes applied:
- added in-app `Functions` tab with detailed function map
- clarified queue semantics and file outputs directly in the app

## Iteration 5 - scroll and control ambiguity

Reviewer comment:
- "If content is clipped, operator trust drops immediately. Also avoid mixed-state toggles that look like checkbox crosses."

Changes applied:
- added vertical scroll containers for `Pipeline` and `Status` tabs
- removed ambiguous `Dual transcribe` checkbox state
- mode now comes only from `Transcribe mode` (`whisper`/`gigaam`/`dual`)
- added explicit helper text: how to run stage 1 recognition only and where to set batch size

## Final simulated verdict

- Information hierarchy: acceptable
- Primary flows discoverability: good
- Long-run operations confidence: good
- Ready for stable runtime usage, with optional future visual polish (icons/adaptive themes)
