# Обратная совместимость

- Старый lead snapshot/rollback остаётся совместимым: `lead_id` сохранён, lead `snapshot_key` не изменён.
- `build_pre_write_snapshot_rows` получил только optional `entity_type/entity_id`.
- Contact writeback стал строже: нижний helper пропускает только три целевых поля ТЗ.
- Deal writeback стал строже: PATCH невозможен без fresh GET anti-clobber.
- Rollback CLI старые аргументы сохраняет; новые `--contact`, `--lead`, `--snapshot` добавлены.
- Live confirmation tokens не менялись.
