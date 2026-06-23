# Wappi bot-safe memory dry-run report, 2026-06-24

Scope: connect the existing bot-safe Customer Timeline memory to the Wappi -> AMO draft loop in read-only/dry-run mode. No AMO, Tallanto, CRM, Wappi send, or live-write action was performed.

Branch/worktree: `codex/wappi-botsafe-memory` in `/Users/dmitrijfabarisov/Projects/Mango_wappi_botsafe_memory`, based on `4caa5eb`.

## Database

Authoritative production DB path, not opened for write:

`/Users/dmitrijfabarisov/Projects/Mango analyse/product_data/customer_timeline/customer_timeline_prod_20260621/customer_timeline.sqlite`

Read-only working copy created through SQLite backup API from a read-only source connection:

`/Users/dmitrijfabarisov/Projects/Mango_wappi_botsafe_memory/.codex_local/wappi_botsafe_memory_probe_20260624/customer_timeline_ro.sqlite`

Both source and copy passed `PRAGMA quick_check = ok`.

## Step 0: Read-only audit

`bot_context_chunks` counts by chunk type and flags:

| chunk_type | allowed_for_bot | requires_manager_review | rows |
|---|---:|---:|---:|
| `mango_call_summary` | 0 | 1 | 71962 |
| `email_message` | 0 | 1 | 22397 |
| `bot_safe_summary` | 1 | 0 | 17856 |
| `customer_history_summary` | 0 | 1 | 13227 |
| `channel_message` | 0 | 1 | 1230 |

`bot_safe_summary` brand tags from structured tags:

| brand tag | rows |
|---|---:|
| `foton` | 1290 |
| `unpk` | 4017 |
| `unknown` | 12549 |

Trash-like bot-safe summaries by marker scan over `text`/`summary` only:

| brand tag | trash-like rows |
|---|---:|
| `foton` | 1105 |
| `unpk` | 3352 |
| `unknown` | 10284 |
| total | 14741 |

`unknown` trash-like share: `10284 / 12549 = 81.95%`.

PII scan over `text`/`summary` only, not over service `record_json`: phone `0`, email `0`, service id `0`.

Wappi loop bridge confirmed in `scripts/run_amo_wappi_draft_loop.py`: it calls `build_bot_safe_crm_context` and reads `TELEGRAM_BOT_SAFE_CRM_CONTEXT`, `TELEGRAM_BOT_SAFE_CRM_CONTEXT_DB`, and `TELEGRAM_BOT_SAFE_CRM_CONTEXT_TENANT`.

## Step 1: Unknown blocker fix

Changed `src/mango_mvp/customer_timeline/bot_safe_runtime_context.py`:

- `unknown` chunks are no longer visible for an active brand.
- `unknown` is no longer copied into prompt relevance tags.
- placeholder/trash summaries are removed before prompt assembly.
- PII in a summary keeps blocking the chunk.

Changed `src/mango_mvp/integrations/draft_loop.py`:

- dry-run journal now records only safe bot-safe metadata: `found`, `allowed_only`, `active_brand`, item count, warnings, and safety booleans.
- raw summaries, item text, customer ids, timeline ids, and service ids are not written to this metadata block.

## Step 2: Identity probe

Probe over mapped non-ambiguous ids against the read-only copy:

| metric | value |
|---|---:|
| unique AMO-link customers available | 7743 |
| sampled contexts | 61 |
| `foton` | 10 |
| `unpk` | 51 |
| ok | 61 |

All 61 sampled contexts had:

- `found=true`
- `allowed_only=true`
- `raw_timeline_events_included=false`
- `customer_profile_included=false`
- `raw_ids_included=false`
- `pii_scan_passed=true`
- no `unknown` literal in the returned context

Safe examples without personal data:

1. `УНПК`: closed/not implemented, interest `онл`, no active next step found.
2. `УНПК`: paid, interest `ЛШ МФТИ`, no active next step found.
3. `УНПК`: prospect, parent feedback and 2026/27 offer, no active next step found.
4. `Фотон`: waiting for payment, interest in July field school.
5. `УНПК`: interest in online physics/math tracks.

## Step 3: Real identity collision

Found a real collision in `identity_links`: one masked AMO contact link maps to 2 customers. The bridge returned:

- `found=false`
- `customer_resolved=false`
- `warnings=["ambiguous_identity"]`
- no bot-safe context

This confirms that ambiguous identity does not leak memory into the prompt.

## Step 4: Isolated Wappi dry-run

Run mode:

- `run_amo_wappi_draft_loop.py --once --dry-run`
- separate `--local-dir`
- separate `--pairs-file`
- separate `--auto-pairs-file`
- separate `--stop-file`
- `--customer-timeline-db` set to the read-only DB copy
- `--customer-timeline-allowed-root` set to the same local probe directory
- no default `~/.mango_local/draft_loop`
- no default `~/.mango_secrets/draft_loop_pairs.json`

Flags:

- `TELEGRAM_BOT_SAFE_CRM_CONTEXT=1`
- `TELEGRAM_BOT_SAFE_CRM_CONTEXT_DB=<read-only copy>`
- `TELEGRAM_BOT_SAFE_CRM_CONTEXT_TENANT=foton`
- `ENFORCE_CANONICAL_PROFILE=1`
- `TELEGRAM_DIRECT_PATH_PILOT_CONFIG=pilot_gold_v1`
- `TELEGRAM_PRESALE_SAFETY=1`
- `TELEGRAM_PRESALE_PII_MEMORY=1`
- `TELEGRAM_PII_RELATION_STOPWORDS=1`
- `TELEGRAM_VERIFIER_HANDOFF_CLAIMS=1`
- `DRAFT_LOOP_AUTO_RESOLVER=0`

Result:

| metric | value |
|---|---:|
| journal rows | 77 |
| `draft_created` | 1 |
| `pair_missing` | 76 |
| bot calls | 1 |
| dry-run | true |
| AMO note written | 0 |
| client messages sent | 0 |

Draft journal bot-safe metadata:

- `found=true`
- `allowed_only=true`
- `source=customer_timeline_bot_context`
- `active_brand=foton`
- `item_count=1`
- `warnings=[]`
- `customer_profile_included=false`
- `raw_timeline_events_included=false`
- `raw_ids_included=false`
- `pii_scan_passed=true`

Draft text scan:

- phone/email/service id: `0`
- `unknown`: `0`
- trash marker: `0`
- route: `bot_answer_self_for_pilot`

Actual Wappi dry-run example where memory was used safely: the draft answered the current schedule question using the current brand context and did not expose ids, phone, email, raw timeline events, or unknown-brand memory.

Additional offline draft probes showed the same safe plumbing, but they are not enough for a production conclusion: when memory lacks exact class/format slots, the generic draft can still ask broad clarifying questions. That remains a semantic risk for Claude review.

## Step 5: NEG set

Real DB probes:

| case | result |
|---|---|
| ambiguous identity | empty context, `warnings=["ambiguous_identity"]` |
| foreign-brand-only memory | empty context, `warnings=["no_brand_scoped_bot_safe_context"]` |
| unknown-only memory | empty context, `warnings=["no_brand_scoped_bot_safe_context"]` |

Synthetic regression tests:

| case | result |
|---|---|
| PII-only summary | blocked |
| flag off | previous behavior: bot-safe context is not injected |
| journal metadata | only safe flags/counts, no raw text |
| DefaultDenyTransport | non-GET still denied |

## Tests

Targeted tests:

`PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_bot_safe_runtime_context.py tests/test_run_amo_wappi_draft_loop.py tests/test_amo_wappi_transport.py`

Result: `26 passed in 1.46s`.

Safe collect-only:

`PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest --collect-only -q`

Result: `3621 tests collected`.

`git diff --check`: clean.

## Changed files

- `src/mango_mvp/customer_timeline/bot_safe_runtime_context.py`
- `src/mango_mvp/integrations/draft_loop.py`
- `tests/test_bot_safe_runtime_context.py`
- `tests/test_run_amo_wappi_draft_loop.py`

## Blockers and risks

- Only one recent Wappi chat in the isolated probe had both a pair and bot-safe memory, so the live Wappi sample is intentionally small.
- Strict brand filtering drops `unknown` memory. This is correct for safety; the audit shows most `unknown` rows are trash-like, but useful unknown rows will also be withheld until re-tagged.
- The bridge is safe, but the downstream draft prompt does not yet have a strong semantic rule that says "do not re-ask known slots"; this should be reviewed separately on a wider dry-run sample.

## Recommendation

`formal_pass`: the read-only bridge, strict brand filter, trash filter, identity collision guard, PII guard, and dry-run isolation are in place and tested.

`semantic_risk`: do not call this production-ready. Keep `TELEGRAM_BOT_SAFE_CRM_CONTEXT` default-off and use it only in controlled Wappi dry-runs until Claude review confirms that drafts consistently use memory without re-asking known facts.
