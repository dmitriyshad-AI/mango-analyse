# Dry-run

- SWAP: выполнен только на локальной копии prod DB, production файл не подменялся.
- CRM: выполнен read-only diff по 3 ready карточкам, AMO write=0.
- Wappi: создан manual pairing workbook, auto-link=0.
- Nightly: выполнен staging run, write_prod_db=false.

Итоговые числа:

- staging timeline_events: 171 818
- staging customer_identities: 20 591
- staging bot_context_chunks: 131 679
- staging derived_signals: 2 003
- CRM export: 66 candidates / 3 ready / 63 blocked
