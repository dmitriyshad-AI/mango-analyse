Semantic review status: not applicable as a client-facing text release.

Relevant semantic checks for this engineering block:
- Microprobe report anonymizes child names as `child_name_N`.
- Microprobe report does not include raw phones.
- LLM cache may contain raw child names by design, but it is stored under ignored local artifact path.
- Raw profile DBs and trace artifacts are under ignored `product_data/customer_profiles/`.

No client-facing Telegram/email/CRM/Tallanto text was released.
