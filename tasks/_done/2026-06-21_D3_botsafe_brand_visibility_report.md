# Отчёт D3: видимость bot-safe памяти по бренду

Дата: 2026-06-21.

Ветка: `codex/etap3-faza1-botsafe-bot`.

Что сделано:

- Фильтр показа bot-safe памяти изменён на правило: показывать `active_brand` + `unknown`, скрывать явный чужой бренд.
- То же правило применено в direct path перед prompt.
- В отчёт Фазы 0 добавлен `brand_source_counts`.
- Добавлены тесты на event-brand, unknown-visible и foreign-brand-excluded.

Метрики на SQLite backup test-copy:

- Source copy: `/tmp/mango_botsafe_brand_visibility_backup_20260621/customer_timeline.sqlite`.
- Copy method: `sqlite3.Connection.backup(source mode=ro)`.
- Source DB на момент копии: size `2572046336`, sha256 `9a394f6b0afa281a8d57122f1045f7bf2ffe753943ce470290a4ddcc32fea2b7`.
- Chunks total: `17856`.
- Brand counts: `foton=1290`, `unpk=4017`, `unknown=12549`.
- Brand source counts: `deal=1224`, `event=4083`, `unknown=12549`.
- Known brand by deal: `1222` customers.
- Known brand by deal+event: `5302` customers.
- Customers with both known brands: `5`.
- Deal/event conflicts: `0`.

NEG:

- `unknown_contains_brand_marker=0`.
- `foton_contains_unpk=0`.
- `unpk_contains_foton=0`.
- `legacy_source_ref_count=0`.
- `duplicate_source_ref_groups=0`.
- `raw_allowed_chunks=0`.

Examples:

- Unknown chunk visible in Foton bot: `customer:00013524f17368a066adce9579252a1c` -> context found, no Foton/UNPK label in text.
- Cross-brand customer as Foton: `customer:3dc3c4606c08d0260327c80b9ac893d7` -> contains Foton summary, no UNPK.
- Same customer as UNPK: contains UNPK summary, no Foton.

Test-copy apply:

- First apply on backup copy: `created=0`, `updated=230`, `duplicate=17626`, `skipped=0`.
- Second apply/idempotency: `created=0`, `updated=0`, `duplicate=17856`, `skipped=0`.

Tests:

- Targeted: `136 passed`.
- Full pytest: `3494 passed, 5 skipped`.

Important note:

- Production DB was not written by this worktree. After the backup copy was created, the production DB mtime/size/hash changed in parallel outside this process. Final integration run should use fresh production DB after D8 next-step extractor is ready.
