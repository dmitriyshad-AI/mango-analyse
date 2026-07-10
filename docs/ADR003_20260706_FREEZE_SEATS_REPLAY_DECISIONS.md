# ADR-003 2026-07-06 freeze/seats/replay decisions

## D-001. Freeze is a snapshot, not deploy approval

Decision: keep `docs/ADR003_LIVE_FREEZE_20260706.md` as a read-only snapshot and explicitly mark swap as blocked until a concrete rollback target is recorded.

Reason: live PID `60227` runs from `Mango_main_intent_ff`, but the screen name points to an older `eb6fa0b` build while the worktree is now at `d0357d79`. CWD alone is not enough to know the previous executable code. A human-run deploy plan must not imply rollback readiness when previous worktree/head/screen are unset.

Audit: independent auditor marked the first draft BLOCKED. The fix makes rollback fields explicit `UNSET` and changes the helper so it refuses same-worktree rollback by default.

## D-002. Do not publish secret file fingerprints in shared freeze docs

Decision: remove short SHA fingerprints of secret files from the freeze document; keep only file mode, size, mtime and key names/shapes.

Reason: hashes are not secret values, but they are unnecessary fingerprints in a document intended for handoff and review.

## D-003. Seats default-open is default-OFF and explicit-env only

Decision: introduce `TELEGRAM_SEATS_DEFAULT_OPEN` as a separate explicit flag, not a pilot profile default.

Reason: the owner approved the business policy, but it still needs local micro-pair, semantic review of the client-safe fact, and M1 focus pair before profile enablement. Default-OFF preserves current deploy path if the policy is not ready before deploy.

## D-004. LLM owns the availability meaning; deterministic code only excludes unsafe cases

Decision: the positive path starts only from `SemanticFrame.requested_action=check_availability` with a valid high-confidence inline frame. Deterministic checks are used only as floors/exclusions for camp/shift, individual, unsupported city, booking operation, paid/P0/brand/payment floors.

Reason: this keeps the ADR-003 boundary: model reads the customer meaning, old markers do not choose the business class. The remaining deterministic checks are safety floors and business exceptions.

## D-005. Default-open response is a template, not free generation

Decision: for regular groups under `TELEGRAM_SEATS_DEFAULT_OPEN=1`, return the fixed text: `Места в регулярных группах есть — идёт набор на 2026/27. Помогу записаться: подскажите класс, предмет и формат.`

Reason: the business policy is intentionally broad, but the bot still must not invent concrete group, shift, date, or remaining seat count. A fixed template minimizes fabrication risk.

## D-006. Output floors stay, with a narrow allowlist for the approved template

Decision: keep `_AVAILABILITY_PROMISE_RE`, reliable-answerer floor, and posthoc SemanticFrame manager gate. Add allowlist only when metadata proves the text came from the `seats_default_open_regular_groups` deterministic template.

Reason: arbitrary model text like “места есть, запишем вас” remains unsafe and must be blocked. The allowlist exists only for the owner-approved regular-groups policy.

## D-007. Route is the autonomy criterion; base safety flags are not

Decision: tests for the seats template assert `route=bot_answer_self_for_pilot` and the `seats_default_open_regular_groups` marker, not absence of `manager_approval_required/no_auto_send`.

Reason: `SubscriptionDraftResult.__post_init__` always prepends base safety flags to every result. Removing that invariant for one feature would be a wider contract change unrelated to the business policy.

## D-008. Replay real-provider requires two explicit switches

Decision: `scripts/run_wappi_replay_exam.py` still refuses to run unless exactly one provider mode is selected. Real provider requires both `--real-provider` and `--allow-llm-calls`.

Reason: replay on scrubbed Wappi data is not a live write, but it is still an external model call over customer-derived text. The operator must opt in explicitly.

## D-009. Replay real-provider accepts only scrubbed chat-only cases by default

Decision: real-provider cases must live under `~/.mango_local/replay_exam/scrubbed/`, pass `pii_signals()`, use brand `foton|unpk`, and use `segment=chat_only` unless a separate CLI flag allows other segments.

Reason: raw dumps must never be sent to the draft provider. `external_context` and `manager_issue_private` are different measurement segments and should not be silently mixed into the first real-provider pilot.

## D-010. Replay provider context must not include manager reference

Decision: `manager_reference` is recorded as available only as a boolean in replay metadata; the text itself is not passed to the draft provider.

Reason: the replay target is “what would the bot draft before seeing the manager answer”. Passing the manager answer into context would make the exam invalid.

## D-011. Real-provider pilot was stopped as latency/infrastructure, not object failure

Decision: do not treat the interrupted local real-provider pilot as a quality verdict. Keep adapter and safety tests as complete, but require a separate short pilot or M1 replay run before semantic acceptance of replay quality.

Reason: existing scrubbed pilot-10 set has 10 dialogs / 31 chat-only cases / leak_count=0, but sequential real-provider execution was still running after several minutes and blocked the main implementation pass. The process tree showed normal read-only `codex` calls, not a live-write risk. The local runner was stopped manually to preserve momentum.

## D-012. Seats allowlist must prove the whole approved result, not metadata only

Decision: `_AVAILABILITY_PROMISE_RE` and posthoc manager gate now allow the seats default-open path only if the whole result matches the approved template: route `bot_answer_self_for_pilot`, exact `SEATS_DEFAULT_OPEN_REGULAR_SAFE_TEXT`, safety flag `seats_default_open_regular_groups`, top-level metadata marker, direct-path metadata marker, and allowlist reason.

Reason: the first implementation trusted two metadata fields and could be forged by arbitrary text like “места есть, запишем вас”. The auditor classified this as BLOCKED. The fix adds negative tests for forged metadata.

## D-013. Seats default-open requires active brand and blocks operations/group-size

Decision: the default-open path requires active brand `foton|unpk`; if `requested_product.brand` is present it must match active brand. Booking/enroll operation phrasing, individual lessons, unsupported cities, camp/shift contexts, and group-size questions stay on the manager/fact path.

Reason: “места есть” is a business default for regular groups, not a license to answer cross-brand, booking, paid, individual, camp, or “сколько человек в группе” questions.

## D-014. Replay pilot-10 is accepted as adapter smoke, not as final quality exam

Decision: local real-provider replay pilot completed on scrubbed v2 set: 31 evaluated turns, 8 evaluated dialogs, `real_subscription_llm` provider on all rows, no live-write process left running after completion. Machine gate improved from 7 failures to 4 after scrubber/gate fixes. Remaining failures are `new_number_unverified` on KB-backed prices/dates/schedules.

Reason: this proves the real-provider adapter can run end-to-end over scrubbed data with raw trace and no live writes. It does not prove final quality of the bot and does not replace the later M1 replay exam.

Follow-up: before the big replay exam, pass retrieved/client-safe fact numbers into `run_machine_gate`; otherwise correct KB-backed numbers will be noisy false positives.

## D-015. Replay scrubber must handle mixed-case names and generic contract words

Decision: pseudonymizer now masks mixed-case Russian name/surname pairs like `Сашу кибирева`. PII detector no longer treats generic “договорные документы” as contract ID; it still detects actual contract-like IDs such as `договор ABC-42`.

Reason: replay pilot v1 exposed two measurement/data issues: a partially scrubbed child name and false `pii_in_bot_text` signals on normal contract wording. Both are measurement issues for replay, not live bot object bugs, and both now have regression tests.
