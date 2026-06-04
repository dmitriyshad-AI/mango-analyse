# Backward Compatibility

- The new behavior is behind `TELEGRAM_Q_COMPOSITE`; default OFF means existing runs keep previous behavior.
- No snapshot, KB, stable_runtime, runner flags, or live integrations were changed.
- Existing partial-yield and thread-memory tests remain green.
- Full pytest passed with the expected KB release path.
