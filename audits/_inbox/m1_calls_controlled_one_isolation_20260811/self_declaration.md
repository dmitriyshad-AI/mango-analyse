# Self declaration

- Code commit: `0027677eba2539968ce6e8183ed5413a9f3de07a`.
- Tree: `dc6d53f170db97041ce86f7874f3a4af6cd6835a`.
- Source/scripts delta: 2,534 additions, 126 deletions.
- Tests delta: 2,512 additions, 2 deletions.
- Новые runtime-файлы: allowlist creator и controlled scope module.
- Новые runtime-зависимости не добавлялись.
- Переиспользованы relocation/cutover authority, process leases, owner-only IO,
  SQL claim state, sequential worker orchestration и audit reports.
- Отвергнута запись общего controlled lineage marker: она разблокировала бы
  service cutover.
- Отвергнута слепая строковая замена путей SQLite.
- Отвергнута автоматическая публикация до человеческой проверки.
- Формальная/семантическая приёмка: PASS.
- Data: synthetic only.
- Business/runtime: STOP.
