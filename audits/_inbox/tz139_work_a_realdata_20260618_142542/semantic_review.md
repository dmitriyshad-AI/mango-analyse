# Semantic Review

Status: `semantic_pass` pending external Claude regreade.

Local semantic checks:

- A `no_exact_phone_match` Tallanto status must not become a strong identity. This is now enforced by test and live-source aggregate count.
- A shared family phone with several Tallanto students must not merge children into one customer. This is now enforced by test and live-source aggregate count.
- Shared AMO contact/lead IDs must create manual-review pressure and ambiguous links, not a merge key.
- Reports must not leak raw phones, ФИО or email; only aggregate counts and hashes are present.

Known remaining semantic control:

- Work D must add read-time old -> new `customer_id` resolution for persisted bot context and derived signals.
