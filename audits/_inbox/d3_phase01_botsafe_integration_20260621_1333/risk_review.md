# Risk Review

Primary risks checked:

- Production DB write risk: blocked by using SQLite backup copy in `/tmp`; production `sha256`, size and `mtime` unchanged.
- PII leak risk: runtime scanner findings `{}` on visible bot-safe chunks.
- Name leak risk in `interest/title`: scan after excluding protected program/org names returned `{}`.
- Brand mix risk: runtime visibility checks found no explicit foreign brand markers for active brand.
- Stale-memory risk: 100 stale `unknown` chunks were retired from bot visibility.

Residual risks:

- Safe phrase allowlist is intentionally narrow. Future program names with two capitalized words can require extending it.
- Financial/legal titles can remain in bot-safe context if they are not PII; P0 output gate must stay enabled.
