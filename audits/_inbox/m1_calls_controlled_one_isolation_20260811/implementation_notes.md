# Implementation notes

## Корневые решения

1. Добавлен отдельный `processing_scope=controlled_1`; `stage_limit=1` не
   считается изоляцией сам по себе.
2. Owner-only allowlist v2 содержит ровно один canonical `source_call_id`, ID
   строки, SHA/размер аудио, tenant, code SHA и host ID.
3. Все claim/recovery/readback пути четырёх стадий повторно применяют точный
   scope; широкие capture/ingest/process/sync/publication пути fail-closed.
4. Оба ASR читают одну приватную owner-only копию проверенного исходного аудио.
5. Оркестратор запускает стадии строго последовательно и выпускает отдельный
   короткоживущий stage-ticket, связанный с PID, flock, allowlist, target,
   провайдером и SHA проверенного cutover manifest.
6. Controlled authority проверяет fresh previous-host proof, active host,
   transferred cursor SHA и неизменность manifest read-only. Общий
   `cutover_cursor_lineage.json` не создаётся; service остаётся STOP.
7. Отчёт сравнивает digest всех нецелевых строк, target row, аудио и реальные
   runtime receipts. Нулевой повтор требует неизменного target digest.
8. Cleanup snapshot возвращает отдельное evidence. Missing/tamper/unlink/rmdir
   дают `status=failed`, `pilot_transition_proven=false`; полный
   `before/stages/after` сохраняется. Cleanup `OSError` не маскирует первичную
   ошибку стадии.

## Реальные точки запуска

- allowlist: `scripts/create_m1_calls_controlled_allowlist.py`;
- readiness: `scripts/probe_m1_calls_access.py --readiness-target controlled-1`;
- будущий разрешённый пилот:
  `scripts/run_mango_calls_pipeline.py ... controlled-one`;
- worker enforcement: `src/mango_mvp/services/controlled_call_scope.py` и
  четыре service-класса.

Команда пилота не выполнялась. Capture, Process A/B, watchdog service mode,
publication и launchd не включались.
