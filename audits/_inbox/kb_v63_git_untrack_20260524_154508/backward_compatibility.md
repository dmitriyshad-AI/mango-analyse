# Backward Compatibility

## Expected Behavior

- Existing local workflows that read current v6.3 generated files continue to work because files remain on disk.
- Future rebuilds of current v6.3 generated outputs will stay untracked because of `.gitignore`.
- Source-driven rebuild remains available through `scripts/build_kb_release_v6_1_team_answers.py`.

## Compatibility Notes

- Old frozen release snapshots remain tracked and were not removed from git.
- Current v6.3 generated outputs will no longer be available directly from git after a fresh clone; they must be regenerated from `kb_release_20260520_v6_3_team_answers_sources/`.
- This matches the requested split: sources in git, regenerable generated outputs outside git.
