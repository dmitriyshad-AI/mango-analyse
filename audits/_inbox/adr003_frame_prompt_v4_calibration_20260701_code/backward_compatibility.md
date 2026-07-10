# Backward Compatibility

- `route`, `draft_text`, `safety_flags`, `manager_checklist` не меняются при paired enrichment: diff = 0.
- `_direct_path_answerability_value` для старого `answerability_self` не изменён; добавлена отдельная нормализация только для `semantic_frame.answerability`.
- `semantic_frame_shadow` остаётся alias того же metadata frame.
- Live/profile не тронуты; изменения проявляются только при включённом shadow-флаге SemanticFrame.

