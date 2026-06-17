# TZ-33 perf flags verify and enable

Date: 2026-06-17

Branch/worktree: `codex/tz33-perf-verify-enable` in `/Users/dmitrijfabarisov/Projects/mango-tz33-perf`.

## Scope

Verified already implemented performance flags and enabled safe defaults. No live AMO/Tallanto writes, no ASR, no Resolve+Analyze, no profile rebuild.

TZ-143 amendment on 2026-06-18: `TALLANTO_BATCH_FETCH` was reverted to default OFF because `live_card_only` changes data composition used by deal-aware compact contexts.

## A. TALLANTO_BATCH_FETCH

Decision: keep default OFF after TZ-143.

Code points:

- `src/mango_mvp/amocrm_runtime/tallanto_api.py`
- `src/mango_mvp/amocrm_runtime/tallanto_context.py`
- `tests/test_tallanto_api.py`

Verified behavior:

- This is not true Tallanto batch-by-id. It is early stop in phone field search plus `live_card_only` skipping blocks unused by the live card.
- Default/OFF keeps the old full scan path.
- Explicit ON produces fewer Tallanto mock calls and the same rendered live card, but it drops `opportunities`, `requests`, and `course_relations` before `compact_contexts`.
- TZ-143 invariant: with default/OFF, both `build_contact_context` and `build_contact_context_by_contact_id` preserve `opportunity_count>0` and `course_relation_count>0` through the real `compact_contexts` consumer.

Residual limitation: if the same phone appears across multiple Tallanto fields and maps to distinct contacts, explicit ON can skip later contacts. Keep default OFF until all consumers of the dropped blocks are audited.

## B. AMO_LEADS_BATCH_FETCH

Decision: enable by default.

Code points:

- `src/mango_mvp/amocrm_runtime/amo_integration.py`
- `src/mango_mvp/amocrm_runtime/deals.py`
- `tests/test_amocrm_deals.py`

Contract check:

- Official amoCRM docs for `GET /api/v4/leads` define `filter` for lead lists.
- Official amoCRM filtering docs define `filter[id]` as `int|array` for leads and mark it as multiple.
- Current implementation sends `filter[id][]=...` through URL encoding, with chunk size 50 and `with=contacts`.

Verified behavior:

- Mock URL includes `filter[id][]=11&filter[id][]=22`, `with=contacts`, and matching `limit`.
- `fetch_leads_batch` preserves input order even if AMO returns leads out of order.
- NEG: OFF resolves by two single `fetch_lead` calls; ON/default resolves via one `fetch_leads_batch` call.
- NEG: OFF, ON, and default ON produce the same candidate list and selected lead.

## C. PROFILE_PHONE_INDEX

Decision: enable by default.

Code points:

- `src/mango_mvp/customer_profile/store.py`
- `src/mango_mvp/customer_profile/crm_summary.py`
- `tests/test_customer_profile_crm_summary.py`

Verified behavior:

- Explicit `PROFILE_PHONE_INDEX=0` preserves old schema without `primary_phone_norm`.
- Default/ON creates `primary_phone_norm`, creates `idx_customer_profiles_phone_norm`, fills normalized phone on profile writes, and preserves lookup.
- NEG: indexed lookup returns the same `profile_id` set as full scan.
- NEG: suffix lookup does not overmatch a phone where the target 10 digits occur before the end of `primary_phone_norm`.

Rebuild note: no profile DB rebuild was run. The column is filled on profile writes/rebuild; production rebuild should happen once together with TZ-32, not separately for TZ-33.

## Tests

Targeted:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q \
  tests/test_tallanto_api.py \
  tests/test_amocrm_deals.py::AmoCrmDealAnalysisTest::test_fetch_leads_batch_uses_amo_filter_id_contract_and_preserves_input_order \
  tests/test_amocrm_deals.py::AmoCrmDealAnalysisTest::test_resolve_target_lead_batch_fetch_is_flagged_and_keeps_selected_lead \
  tests/test_customer_profile_crm_summary.py
```

Result: `21 passed in 0.55s`.

Full pytest:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q
```

First full run had one unrelated flaky failure in `tests/test_productization_mail_archive.py::test_mail_phone_lift_preview_lifts_manual_messages_from_text_phones`: the test searches for literal `"999"` in whole JSON and hit microseconds in `generated_at`.

The failed test rerun passed: `1 passed in 0.53s`.

Second full run passed: `3328 passed, 5 skipped, 1 warning in 49.71s`.

TZ-143 full run passed: `3329 passed, 5 skipped, 1 warning in 52.08s`.

Warning: local urllib3/OpenSSL warning from system Python; unrelated to TZ-33.

## Outcome

Final flag defaults after TZ-143:

- `TALLANTO_BATCH_FETCH`: default OFF.
- `AMO_LEADS_BATCH_FETCH`: default ON.
- `PROFILE_PHONE_INDEX`: default ON in both `store.py` and `crm_summary.py`.
