# TZ-16 final report: profiles v7 and rerun tail

Date: 2026-06-12
Branch: `codex/tz16-d4-profiles-v7-rerun-tail`

## Commits

- `463da53c` - TZ16 step1 cleanup replant worktree
- `00e8d442` - TZ16 step2 rebuild profiles on v7 summaries
- `f93249c4` - TZ16 step3 size rerun tail
- `a5ea83ba` - TZ16 step4 protect live dialogues from autoresponder

## Step 1

- Removed old clean worktree: `/Users/dmitrijfabarisov/Projects/Mango-analyse-tz14-replant`
- Main dirty project folder was not switched, cleaned, or touched.
- Current worktree: `/Users/dmitrijfabarisov/Projects/Mango-analyse-tz16`

## Step 2

Output folder, ignored by git:

- `product_data/customer_profiles/tz16_profiles_v7_20260612/`

Profile rebuild:

- profiles built: `18399`
- fields written: `190614`
- superseded fields: `72791`
- unmatched calls: `1053`
- full build seconds: `38.408`
- idempotence: content signatures equal on repeated build
- `tz12_working_batch3` unchanged: `true`
- source hash before/after: `1496cb86bce139f719690d018e79f11e08842e033b5a95cb96403e77684edc22`

Profile metrics, old -> new:

- profiles: `18399 -> 18399`
- profiles with 2+ children: `4471 -> 4411`
- merge-candidate profiles: `807 -> 824`
- active fields: `116957 -> 117823`
- fields total: `189483 -> 190614`
- superseded fields: `72526 -> 72791`

Key coverage, old -> new:

- parent_name: `6185 -> 6291`
- child_name: `6561 -> 6623`
- grade: `7317 -> 7333`
- subject: `8935 -> 8921`
- format: `9148 -> 9168`
- target_product: `8995 -> 9023`
- next_step: `9895 -> 9941`
- objection: `11269 -> 11300`
- child_slot_merge_candidate: `807 -> 824`

Analyze counters:

- valid analysis JSON rows: `65939`
- v7 rows: `22679`
- non-v7 rows: `43260`
- blacklist ids loaded/present: `77/77`
- blacklist ids with v7: `0`
- blacklist preserved old: `77`

## Step 3

Customer rerun zone:

- active AMO customers: `1303`
- strong Tallanto student customers: `7298`
- union customers: `7965`
- calls in zone: `46187`
- calls missing in canonical master DB: `0`

v7 first slice:

- package ids total: `22756`
- package ids inside zone: `16853`
- zone calls current v7: `16797`
- zone calls current old: `29390`
- blacklist inside zone preserved old: `56`

Tail:

- old-summary calls: `29390`
- non-blacklist old-summary calls: `29334`
- old `>=60` sec including blacklist: `3495`
- old `>=60` sec excluding blacklist: `3439`
- same-filter second slice left (`>=2025-06-01`, `>=60`, excluding blacklist): `0`

Tail split:

- recent `>=2025-06-01` and `>=60` sec, blacklist: `56`
- recent `>=2025-06-01` and `>=60` sec, non-blacklist: `0`
- recent `>=2025-06-01` and `<60` sec: `20749`
- old `<2025-06-01` and `>=60` sec: `3439`
- old `<2025-06-01` and `<60` sec: `5146`

Second-batch estimate:

- useful old `>=60` sec excluding blacklist: `3439` calls, `15682102` transcript chars
- estimated wall time by observed throughput: `4.747` hours
- all non-blacklist old tail: `29334` calls, observed-throughput estimate `40.494` hours

## Step 4

Changed behavior:

- long live dialogue with third-party organization / IVR-like words is not forced into `non_conversation`;
- v7/full prompt explicitly blocks the long-dialogue -> autoresponder shortcut;
- true voicemail/short IVR/virtual secretary controls remain blocked.

Real read-only controls:

- blacklist sample `12617`, `14115`, `14327`, `15112`, `16146`: `5/5` deterministic `service_call`, `0/5` forced `non_conversation`
- true autoresponder controls `15717`, `16565`, `24790`: `3/3` remain forced `non_conversation`

LLM microprobe:

- runner: `codex_cli`
- model: `gpt-5.4-mini`
- profile: `full`
- `llm_calls_total = 5`
- elapsed seconds: `89.688`
- `5/5` normalized to `service_call`
- `5/5` prompt version `v7`
- raw transcripts/summaries written: `false`
- sanitized ignored artifact: `product_data/customer_profiles/tz16_profiles_v7_20260612/step4_blacklist_microprobe.json`

Semantic note: the false-autoresponder class is fixed on the 5-call microprobe, but one case still had `target_product_present=true`. Full 77-call rerun should stay gated by a reviewed small batch or an additional semantic check.

## Anonymized Profile Examples

| example_id | profile_hash | has_phone | events | active_fields | fields | child_slots | brands |
|---|---|---:|---:|---:|---|---:|---|
| profile_example_1 | `bac8255934dc` | true | 409 | 0 | none | 0 | none |
| profile_example_2 | `55d79fc10eb7` | true | 349 | 40 | child_name, format, grade, next_step, objection, subject, target_product | 7 | foton |
| profile_example_3 | `62a3c3862801` | true | 313 | 6 | format, grade, next_step, objection, subject, target_product | 1 | unknown |
| profile_example_4 | `e29dcec3c150` | true | 261 | 6 | child_name, child_slot_merge_candidate, next_step, objection, target_product | 2 | unknown |
| profile_example_5 | `4121d88682dc` | false | 246 | 0 | none | 0 | none |

## NEG

- `tz12_working_batch3` hash unchanged.
- Repeated profile build is idempotent.
- Generated customer profile DBs and microprobe artifacts are ignored by git.
- Active AMO zone excludes closed/won deals.
- Ambiguous Tallanto student links do not enter strong Tallanto zone.
- Blacklist calls are counted separately and not silently rerun.
- Synthetic long third-party business dialogue is not forced as IVR.
- Pure short IVR, voicemail and virtual secretary still force `non_conversation`.
- Real canonical autoresponder controls still force `non_conversation`.
- No AMO/Tallanto/CRM writes.
- No ASR and no Resolve+Analyze.

## Tests

- Step 2 targeted: `24 passed`
- Step 3 targeted: `4 passed`
- Step 4 targeted: `35 passed, 1 skipped, 22 deselected`
- Full pytest: `3064 passed, 5 skipped, 1 warning in 45.54s`

## LLM Calls

- `analyze_microprobe`: `5`
- all other roles: `0`
- `llm_calls_total`: `5`

## Residual Risk

Formal pass is green. Semantic pass is `PASS_WITH_NOTES`: the microfix prevents the dangerous false-autoresponder collapse, but the microprobe still showed one possible over-extraction of target product. Do not run the full 77-call rerun until Dmitry approves the next reviewed batch.
