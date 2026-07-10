# Backward Compatibility

Поведение бота не меняется:
- нет изменений в `src/mango_mvp/channels/subscription_llm_parts/provider.py`;
- нет изменений в `direct_path.py`;
- нет изменений в флагах и профиле;
- нет изменений в KB/runtime данных.

Совместимость отчётов:
- `schema_version` поднята до `adr003_frame_calibration_queue_v2_2026_07_02`;
- старые поля сохранены;
- добавлены новые поля `current_handoff_queue`, `next_autonomy_workstream`, `source_fact_candidate_*`, `source_alignment_*`.

Если внешний читатель ожидает только v1, он продолжит видеть старые summary-поля, но для F2x-регрейда нужно читать новые поля.
