# Backward Compatibility

Изменён только post-hoc SemanticFrame prompt, который работает за
`TELEGRAM_SEMANTIC_FRAME_POSTHOC_SHADOW`.

Флаг остаётся default-OFF и не включён в профиль. Inline direct-path prompt,
route/text, P0, брендовые правила и live runtime не изменены.

Paired no-op report подтвердил:

- compared_turns: 241;
- route_text_diff_count: 0;
- input_diff_count: 0;
- on_non_frame_total: 0;
- hard_gate_failures: 0.
