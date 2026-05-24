# KB v6.3 Git Untrack Audit

## Scope

- Task: remove only regenerable current v6.3 knowledge-base outputs from git tracking.
- Protected scope: `src/mango_mvp/` was not edited, staged, or committed.
- Kept under git: `product_data/knowledge_base/kb_release_20260520_v6_3_team_answers_sources/` and `release_manifest.yaml`.
- Restored frozen old release deletions under `product_data/knowledge_base/` before untracking current v6.3 outputs.

## Actions

- Added exact `.gitignore` entries for current v6.3 generated outputs:
  - `kb_release_20260520_v6_3_team_answers/`
  - `kb_release_20260520_v6_3_team_answers_bot_pack/`
  - `kb_release_20260520_v6_3_team_answers_employee_pack/`
  - `kb_release_20260520_v6_3_team_answers_handoff_for_claude_and_team/`
  - `kb_release_20260520_v6_3_team_answers_smoke_not_run/`
- Ran `git rm -r --cached --ignore-unmatch` for those five generated output directories.
- Confirmed all five directories still exist on disk after untracking.
- Confirmed generated v6.3 paths have `0` tracked files after untracking.
- Confirmed v6.3 source directory still has tracked files.

## Regeneration Check

- First audit build showed the on-disk generated outputs were stale relative to current `*_sources`.
- Rebuilt the current v6.3 generated outputs in place from `*_sources`.
- Built a separate audit copy under `regenerated/`.
- Exact file comparison differs only in build timestamps and output-root paths.
- Normalized comparison result: `normalized_one_to_one=true`, `files_compared=69`.

## Audit Artifacts

- `build_output.json`: first audit build result.
- `compare_diff_qr.txt`: initial comparison before in-place regeneration.
- `compare_after_inplace_diff_qr.txt`: comparison after in-place regeneration.
- `normalized_compare_report.json`: normalized one-to-one comparison result.
- `normalized_compare_hashes.jsonl`: normalized file hashes.
- `test_output.txt`: targeted pytest and collect-only output.
