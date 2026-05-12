# KC sales process source, 2026-05-12

Source DOCX: `/Users/dmitrijfabarisov/Claude Projects/Foton/База знаний КЦ.docx`

Extracted text artifact: `/Users/dmitrijfabarisov/Projects/Mango analyse/stable_runtime/kc_knowledge_base_20260512/kc_knowledge_base_extracted.md`

## Why this file matters

This DOCX is now treated as a business-process source for CRM/AMO enrichment, not just as a generic knowledge base.

It describes how the call center currently works with:

- AMO leads, contacts, deals, duplicate contacts, and active-deal checks.
- Tallanto synchronization and AMO/Tallanto linking.
- Recording students into groups/classes through the AMO widget.
- The distinction between contact communication history and deal feed.
- Perspective/next-year deals and follow-up tasks.
- Feedback calls, payment checks, early-booking offers, and next-year transitions.
- Product facts for courses, intensive programs, and summer schools.

## Important business rules extracted

### Active deal before new action

Before contacting or processing a lead, the manager must check whether the client already has active deals.

If an active deal exists for the same product/interest:

- do not create a duplicate deal;
- pass the lead/context to the manager responsible for the existing deal;
- continue work in the relevant deal, not only in the contact card.

Implication for Mango Analyse: contact-level history is not enough. The next major step must be deal-aware writeback.

### Lead funnel before B2C deal

The DOCX states that contacts should start in the lead funnel. Only after interest is identified and there is something to offer should they be moved into B2C deals.

If a B2C deal was created directly and synchronization broke, the process is to return it to the lead funnel first and then move it back or close it correctly.

Implication for Mango Analyse: AI should not blindly recommend deal movement unless it knows the current funnel/stage and synchronization status.

### Contact history vs deal feed

Contact communication history should contain only concise key information useful to branches:

- product of interest;
- group level;
- preferred schedule;
- important client inputs.

Deal feed can contain operational comments, reminders, and temporary agreements.

Implication for Mango Analyse:

- `Авто история общения` should stay compact and branch-safe.
- More operational AI recommendations should go into deal-level fields/feed/tasks later, not only into the contact history.

### Perspective for next year

If a client may become relevant later, the expected process is:

- create/keep a B2C deal;
- move it to a perspective stage;
- add a note with the essence of the agreement;
- create a follow-up task.

The client should not be recorded into a group/class until they decide where they want to go.

Implication for Mango Analyse: cases like “перенести на 2026/2027” must not recommend the old product/old campaign as the next action.

### Payment and enrollment

For feedback and next-year calls, the manager checks Tallanto payment status. When the deal reaches payment collection, the student can be recorded into the relevant AMO group via the Tallanto widget.

Implication for Mango Analyse: if AMO/Tallanto already shows payment or receipt, AI must not recommend “send payment link” or estimate the deal as merely warm.

### Duplicate contacts

Duplicate contacts must be merged carefully by phone/email, but siblings/relatives must not be merged accidentally.

Implication for Mango Analyse:

- duplicate/multi-contact rows must remain blocked from live writeback until manually resolved;
- the program should help produce duplicate-resolution queues, not auto-merge.

## Impact on current roadmap

This source confirms the ROP feedback:

Contact-level AI writeback is useful, but the daily workflow of the call center happens mainly in deals.

The next substantial product step should be:

1. Build a deal-aware context layer.
2. Match calls to the right AMO deal when possible.
3. Write AI summaries and next actions into deal-level fields/feed for the relevant deal.
4. Keep contact-level history as a compact cross-deal summary.
5. Add guards for paid/closed/lost/perspective/current-campaign conflicts.

## New quality gates to add before deal writeback

- Do not recommend payment/follow-up if the deal is already paid or closed.
- Do not recommend old campaign/product if the client was moved to next academic year.
- Do not write into a deal when several plausible active deals exist unless confidence is high.
- Do not duplicate the same history block across contact and deal fields.
- Separate contact summary, deal summary, and operational next step.
- Keep duplicate-contact cases blocked until human merge/resolution.

## Open product question

For stage 1, we can write deal-aware AI fields without creating AMO tasks.

For a later stage, we should decide whether AI may create AMO tasks automatically. Current safe answer: no, AI only recommends; task creation should be a separate L2/L3 action with approval policy.
