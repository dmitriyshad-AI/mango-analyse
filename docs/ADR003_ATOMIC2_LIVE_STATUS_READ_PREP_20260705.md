# ADR003 atomic #2 prep: live_status_read profile + legacy seats stem removal

Status: prepared only. Do not apply before M1 final pair raw regrade and Dmitry explicit "yes".

Current source revision used for this prep: `4e26c5b22126cc782e19ce1ecfbd0a6e5ed7fb1a`.

## Objective

After the final M1 pair proves `live_status_read/conversation_intent_plan`, do one atomic change:

1. Add `live_status_read` to the pilot profile default semantic reading classes.
2. Add `live_status_read/conversation_intent_plan` to the pilot/default apply configuration.
3. Remove the legacy input live-availability stem/facet decision from `conversation_intent_plan._asks_live_availability`.
4. Keep all safety floors:
   - P0/high-risk/payment/refund floors;
   - output availability promise floor;
   - fail-closed floor when frame is absent/invalid and old plan still sees live availability;
   - manager checklist "do not promise a seat before checking".

This is the second "minus-lasagna" deletion. It must not touch Ж3/Ж4 (`reask_read`, `roles_read`), because both are confirmed NO-GO for apply.

## Evidence Already Present

- `docs/ADR003_ETAP_T_DECISIONS.md`:
  - D-031 defines `live_status_read/conversation_intent_plan` apply semantics.
  - D-032 records deletion #1 and explicitly says `_asks_live_availability` was not touched yet.
  - D-033/D-034 keep Ж3/Ж4 trace-only.
  - D-035 records multi-class/ON-first runner support.
- Local Ж2 micro-pair was regread green by Fable: 16 clean applications, lawful fail-closed, no route/text danger found.
- Final M1 package must be the latest `adr003_final_livestatus_pair_<source-sha>_20260705`
  Yandex folder for this branch, with `SOURCE_HEAD.txt` matching the bundle HEAD
  and ON = `live_status_read` only.

## Exact Candidate Edits After Approval

### 1. Profile default reading classes

File:
`src/mango_mvp/channels/subscription_llm_parts/semantic_reading.py`

Current line:

```python
PILOT_PROFILE_DEFAULT_READING_CLASSES = "sense_seats,slots_gsf,off_topic,intent_actions"
```

Planned line:

```python
PILOT_PROFILE_DEFAULT_READING_CLASSES = "sense_seats,slots_gsf,off_topic,intent_actions,live_status_read"
```

Reason: profile should always compute the new live-status reader once M1 proves it.

### 2. Profile default apply

Current state: there is no profile default for `TELEGRAM_READING_APPLY_CLASSES`; apply is explicit-env only.

Planned implementation must be minimal and auditable:

- introduce a profile default apply list for pilot profile only, or equivalent support-layer helper;
- default must include exactly:
  `live_status_read/conversation_intent_plan`;
- explicit `TELEGRAM_READING_APPLY_CLASSES` must keep priority over profile default;
- without `pilot_gold_v1`, default apply must remain empty.

Required unit matrix:

- pilot profile + no explicit apply -> contains `live_status_read/conversation_intent_plan`;
- pilot profile + explicit `TELEGRAM_READING_APPLY_CLASSES=0` or empty override -> disables default apply;
- no pilot profile -> apply set empty;
- unrelated classes (`reask_read`, `roles_read`, `route_templates`) are not profile-applied.

### 3. Remove legacy seats stem/facet understanding

File:
`src/mango_mvp/channels/conversation_intent_plan.py`

Candidate function:
`_asks_live_availability(text, previous_product_family, product_family)`

Current rule uses regex/markers:

- negation guard for "не про места";
- payment terms guard;
- availability words: `налич`, `брон`, `заброни`, `свободн`, `остал`;
- place-state regex around `место/места`;
- fix-place regex around `закреп/оформ/запис` + `место/места`;
- camp context requirement.

Planned deletion must not blindly remove all behavior unless replacement is active. The safer shape:

- when `live_status_read/conversation_intent_plan` is active and frame is valid, `SemanticFrame.requested_action=check_availability` supplies the live-status decision;
- when frame is absent/invalid, keep fail-closed manager floor via existing `intent_actions` fallback (`conversation_plan_live_availability_floor`) if the old plan still sees live availability;
- do not remove output promise regex `_AVAILABILITY_PROMISE_RE` or reliable answerer availability floor.

Implementation note: if `_asks_live_availability` remains only as a fail-closed fallback, rename/mark it as fallback, and remove it from the normal "source of meaning" path. The deletion credit should count only if normal `primary_intent` no longer comes from this regex when frame is valid.

## Required Tests Before Commit

Targeted:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q \
  tests/test_subscription_llm_draft_provider.py \
  tests/test_adr003_semantic_reading_e3_runner.py \
  tests/test_adr003_regex_understanding_moratorium.py
```

At minimum, add or update tests for:

- profile default reading includes `live_status_read`;
- profile default apply includes only `live_status_read/conversation_intent_plan`;
- explicit env override can disable profile apply;
- `Есть места на смену 6-17 июля?` still becomes manager/check-live via frame;
- `Место занятий где?` does not become live availability;
- `это справка, не бронирование` does not trigger live-status demote;
- paid/no-place and paid/no-access contexts stay manager/floor;
- output text promising places without live fact is still blocked.

Local smoke after tests:

```bash
REV_LABEL=adr003_atomic2_local_smoke_$(git rev-parse --short HEAD) \
RUN_ORDER=ON_FIRST \
TARGET_READING_CLASSES= \
TARGET_APPLY_CLASSES= \
OUT=runs/adr003_atomic2_profile_smoke_$(git rev-parse --short HEAD)_$(date +%Y%m%d_%H%M%S) \
bash scripts/run_adr003_semantic_reading_e3_paired.sh --dry-check
```

The smoke must prove profile-as-is emits `live_status_read` without manual target env.

## Stop Conditions

Stop and ask before committing if any of these happen:

- M1 final pair is not green by raw regrade;
- any P0/payment/refund/brand/manager-only route is lowered;
- any seats question becomes `bot_answer_self_for_pilot` without manager/check-live floor;
- output availability promise floor weakens;
- profile default accidentally enables Ж3/Ж4 or `route_templates` apply;
- moratorium snapshot changes outside the intended legacy deletion and budget update.

## Non-Goals

- Do not fix Ж3 reask or Ж4 money/tax/recording roles here.
- Do not clean frozen monolith code here.
- Do not touch live runtime, Wappi, Telegram, AMO, Tallanto, CRM, or push.
