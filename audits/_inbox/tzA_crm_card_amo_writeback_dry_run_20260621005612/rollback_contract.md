# Snapshot / rollback

- При доступном AMO read-only контексте dry-run создаёт:
  - `pre_write_snapshot.jsonl`;
  - `pre_write_snapshot.csv`;
  - `rollback_manifest.json`;
  - `write_journal_dry_run.jsonl`.
- Snapshot строится через D7 `build_pre_write_snapshot_rows`.
- Anti-clobber строится через D7 `pre_patch_write_decisions`.
- Реальный rollback не нужен: live write не выполнялся.
- Если будущий live будет разрешён, rollback использовать только отдельной командой и отдельным подтверждением.
