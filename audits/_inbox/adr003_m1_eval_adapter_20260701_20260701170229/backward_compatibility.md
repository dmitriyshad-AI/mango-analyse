# Backward Compatibility

- Existing `client-mode fake` and `client-mode codex` behavior is unchanged.
- `scripted` is opt-in.
- `initial_history_lines` only affects personas that explicitly provide it; existing shipped persona sets do not use this field.
- `dynamic_turns.csv` gains two telemetry columns; existing columns stay in place.
- The builder keeps the old sanitized Wappi case file format and adds a separate runner-compatible M1 scenario file.
