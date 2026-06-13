Semantic review for D1/D2

D1:
- OFF keeps the full Tallanto phone-field scan: mock count 15 calls.
- ON returns after the first hit: mock count 1 call.
- Full live context fetches opportunities, requests, finances, course_relations, class_relations, abonements, and classes once each.
- ON live_card_only skips only opportunities, requests, and course_relations; live card output remains byte-identical in the fixture.

D2:
- OFF keeps two per-lead fetch_lead calls for two embedded leads.
- ON uses one fetch_leads_batch call and zero fetch_lead calls.
- Selected lead is identical between OFF and ON.
- Batch HTTP contract is verified through amo_api_request URL encoding: filter[id][]=11&filter[id][]=22, with=contacts, limit=2.

Residual semantic risk:
- Tallanto true server-side class batch API was not introduced because the local client has no confirmed get_entries_by_ids contract. This block still reduces requests on the live-card path without changing live-card content.
- AMO batch is verified by local HTTP contract and mocks only; no live AMO read was performed.
