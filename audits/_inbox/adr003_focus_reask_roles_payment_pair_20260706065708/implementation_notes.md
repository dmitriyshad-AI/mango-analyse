# Что сделано

- Сужен коридор `roles_read/refund_tax`: `route=manager_only` больше не считается самостоятельным доказательством ложного refund. Tax-шаблон применяется только для подтверждённого tax/non-refund plan и refund-похожего выхода.
- Добавлена регрессия `test_direct_path_roles_read_apply_does_not_clear_unrelated_manager_only`: tax-plan рядом с payment dispute не снимает `manager_only`.
- Собран фокусный M1-набор `adr003_focus_reask_roles_payment_20260706.jsonl`: 20 существующих PaymentFix-персон + 15 персон на `reask_read`, `roles_read`, #16.
- Решения D-042/D-043 записаны в `docs/ADR003_ETAP_T_DECISIONS.md`.

# Как проверялось

- `bash -n scripts/run_adr003_semantic_reading_e3_paired.sh`
- `pytest roles_read_apply subset`: 3 passed
- `pytest semantic_reading/e3_runner/moratorium/report`: 94 passed
- `pytest --collect-only`: 4150 tests collected
- Локальный dry-check пары:
  - `VALID_E3_ON`: 2 dialogs / 4 turns / eligible_frame_rate=1.0 / hard=0
  - `VALID_E3_B`: 2 dialogs / 4 turns / eligible_frame_rate=1.0 / hard=0

# Что осталось

- Полную B/ON пару не гонял локально: это задача M1, чтобы не тратить локальную машину и получить сравнение на одном железе.
- После M1 нужен raw-регрейд диалогов, особенно route/text diffs, P0/money, brand, numbers, #16.
