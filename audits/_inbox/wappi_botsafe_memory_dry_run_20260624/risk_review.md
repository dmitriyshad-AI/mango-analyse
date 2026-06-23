# Risk Review

External write risk: low. All Wappi execution was `--dry-run`; no AMO/Tallanto/CRM write was performed, and no client message was sent.

Production DB write risk: low. The authoritative timeline DB was copied through SQLite backup API from a read-only source connection, and probes used the local ignored copy.

PII risk: controlled. Reports contain counts, hashes, and sanitized examples only. Local raw probe artifacts are under ignored `.codex_local/`.

Business risk: medium until wider dry-run review. The memory bridge is stricter and safer, but prompt behavior still needs semantic review on more live-shaped Wappi cases.

Operational risk: medium-low. `TELEGRAM_BOT_SAFE_CRM_CONTEXT` remains default-off; enabling requires explicit environment flag and DB path.
