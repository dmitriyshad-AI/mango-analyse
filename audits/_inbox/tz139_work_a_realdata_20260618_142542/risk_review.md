# Risk Review

- `formal_pass`: tests are green.
- `semantic_pass`: pending Claude regreade. Do not treat this as final acceptance.
- The live SQLite was not rewritten; after-values are from in-memory Work A recomputation over the matching source snapshot.
- The missing manifest path is a source hygiene issue, but the effective `master_contacts_ru.csv` SHA matches the manifest SHA.
- Read-time resolver old -> new `customer_id` is not implemented here by design; it remains Work D.
- AMO shared contact/lead non-merge is verified through source aggregate policy and tests, not through a full regenerated production SQLite.
- No live write path was exercised.
