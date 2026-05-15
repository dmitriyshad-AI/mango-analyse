# Readback Plan

После будущего live-микропилота на 1-5 сделках нужно:

1. Проверить наличие:
   - `pre_write_snapshot.jsonl`;
   - `pre_write_snapshot.csv`;
   - `rollback_manifest.json`;
   - `live_write_report.csv`;
   - `live_write_report.json`.
2. Запустить существующий readback gate по `deal_stage6_writeback_report.csv`.
3. Сверить AMO readback с `preview_payload`.
4. Проверить, что нет:
   - missing target fields;
   - readback mismatch;
   - CRM text quality blockers.
5. Только после успешного readback решать, расширять ли микропилот.

Если readback показывает проблему, сначала запускать rollback dry-run. `--apply` только после отдельного подтверждения.
