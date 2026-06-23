# Backward Compatibility

- Добавлены новые поля к JSON/JSONL-фактам и metadata чанков: `venue`, `program_kind`, `venue_inference_source`, `program_kind_inference_source`, `scope_axes_schema_version`.
- Существующие поля не удалялись.
- При выключенном будущем флаге код, который игнорирует новые поля, должен сохранить прежнее поведение.
