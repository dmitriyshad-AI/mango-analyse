TZ22 D7 block D1/D2: Tallanto and AMO batch/read optimization

Scope:
- D1: TALLANTO_BATCH_FETCH default OFF.
- D2: AMO_LEADS_BATCH_FETCH default OFF.

D1 implementation:
- Tallanto phone search returns after the first found contact only when TALLANTO_BATCH_FETCH=1.
- Added live_card_only mode to TallantoApiClient context builders.
- build_live_tallanto_context passes live_card_only only when TALLANTO_BATCH_FETCH=1.
- In live_card_only mode, opportunities, requests, and course_relations are not fetched because build_tallanto_live_card does not use them.
- Finances, class_relations, abonements, and classes are still fetched because live_card uses them.

D2 implementation:
- Added fetch_leads_batch() near fetch_lead() in amo_integration.py.
- Contract uses GET /api/v4/leads with params filter[id][]=... and with=contacts.
- resolve_target_lead uses fetch_leads_batch only when AMO_LEADS_BATCH_FETCH=1.
- OFF path keeps the old per-lead fetch_lead loop.

Out of scope:
- No live AMO or Tallanto requests.
- No analyze/ASR/heavy reruns.
