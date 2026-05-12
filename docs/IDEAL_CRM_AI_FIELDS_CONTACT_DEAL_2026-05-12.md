# Ideal CRM AI fields for contact and deal, 2026-05-12

This document defines the target field design for AMO enrichment after adding Tallanto, future messenger/email, and deal-aware context.

## Core principle

Contact card = full cross-channel client/family memory.

Deal card = only context and next action for one specific sale/deal.

If a fact belongs to the student/family overall, it belongs in contact-level AI fields. If it changes what the manager should do in one AMO deal, it belongs in deal-level AI fields.

Each AI field must answer its own question. Do not duplicate the same paragraph, sentence, or full set of facts across several fields. Repetition is a product defect because a manager has to reread the same information several times.

Field uniqueness policy:

- summary fields: state and conclusion only;
- history fields: timeline only, without copying the summary paragraph;
- next-step fields: one concrete action only, without history;
- priority/status fields: short label only;
- Tallanto fields: study/finance facts only;
- warning/risk fields: only the reason to be careful.

## Contact-level AI fields

### Minimum production fields

| Field | Type | Purpose |
|---|---|---|
| `AI-краткая сводка клиента` | textarea | Short summary of who the client/student is, current state, and key history |
| `AI-история общения` | textarea | Compact timeline across calls, AMO, Tallanto, and later TG/email |
| `AI-активные сделки клиента` | textarea | List of active/relevant AMO/Tallanto deals with product, status, owner, next action |
| `AI-учебный контекст Tallanto` | textarea | Groups, subjects, classes, teachers, current/historic enrollment |
| `AI-финансы Tallanto` | textarea | Payment, abonement, write-off, refund/debt, remaining package status |
| `AI-важные риски клиента` | textarea | Duplicates, several children, several phones, conflict between AMO/Tallanto, low-confidence match |
| `AI-дата обновления` | text/date | When AI context was last generated |
| `AI-источники контекста` | text | Which sources were used: calls, AMO, Tallanto, TG, email |

### Current contact fields already used

| Existing field | Keep / replace |
|---|---|
| `Авто история общения` | Keep as current compact contact history, later migrate into `AI-история общения` semantics |
| `Последняя AI-сводка` | Keep as latest important touch summary |
| `AI-рекомендованный следующий шаг` | Keep, but only for contact-level action when no clear deal exists |
| `AI-приоритет` | Keep, but later distinguish contact priority vs deal priority |
| `Статус матчинга` | Keep as technical matching status |

## Deal-level AI fields

### Minimum production fields

| Field | Type | Purpose |
|---|---|---|
| `AI-сводка по сделке` | textarea | What is happening in this exact deal |
| `AI-история по сделке` | textarea | Recent relevant touches for this deal only |
| `AI-рекомендованный следующий шаг` | textarea | Concrete manager action |
| `AI-дата следующего касания` | text/date | Recommended follow-up date |
| `AI-фактический статус сделки` | text/select | AI-read status: new interest, thinking, waiting materials, waiting payment, paid, lost, next-year perspective |
| `AI-приоритет сделки` | text/select | hot/warm/cold/service/review |
| `AI-актуальные возражения` | textarea | Current blockers only, not historic filler |
| `AI-основание рекомендации` | textarea | Why AI recommends this step |
| `AI-качество привязки к сделке` | text/select | high/medium/low/manual-review |
| `AI-предупреждение по сделке` | textarea | Payment/closed/duplicate/stale/wrong-deal warning |
| `AI-Tallanto статус по сделке` | textarea | Payment, abonement, group/class/write-off status relevant to this deal |
| `AI-дата обновления сделки` | text/date | When this deal context was last refreshed |

### Current lead fields already present in AMO

The following fields already exist in AMO leads/deals and can support the first MVP:

| Existing field | Current status | Note |
|---|---|---|
| `AI-вердикт по закрытию` | exists | Old reopen/premature-close task; may be reused only for closure review |
| `AI-risk: premature close` | exists | Old risk field; not ideal for all deals |
| `AI-основание вердикта` | exists, textarea | Can serve as explanation field in MVP |
| `AI-рекомендованный следующий шаг` | exists | Should be textarea for longer instructions if possible |
| `AI-дата следующего касания` | exists | OK |
| `AI-сводка по сделке` | exists, textarea | Core deal field |

## Context sources to include

### Already available / partially available

- Calls: ASR + Resolve + Analyze + phone-chain history.
- AMO: contacts, leads/deals, notes/tasks, field values, pipeline/status.
- Tallanto: contacts, communication history, interests, opportunities, requests, finances, abonements, groups, classes, write-off visits report.

### Future sources

- Telegram messages.
- Email threads.
- SMS / web chat if available.

## Tallanto blocks to summarize

### Study block

- active groups;
- historic groups;
- subjects;
- class schedule;
- teacher;
- branch;
- current/historic class relations;
- actual write-offs by class.

### Finance block

- incoming payments;
- invoices;
- abonements;
- remaining visits/hours;
- write-offs for visits;
- refunds/debts;
- receipt status when available.

### Sales relevance block

- already paid/attending;
- paid but not started;
- active abonement near exhaustion;
- old student returning;
- product mismatch between AMO deal and Tallanto actual attendance.

## Quality rules before writing into deals

- Do not write a deal recommendation when there are several plausible active deals and no high-confidence match.
- Do not recommend payment if Tallanto/AMO shows payment, receipt, active abonement, or write-off for the same product.
- Do not treat an active student as a cold lead.
- Do not recommend an old campaign if the client was moved to next academic year/perspective.
- Do not duplicate full contact history inside every deal.
- Do not duplicate the same manager-facing text across summary, history, next-step, Tallanto, and risk fields.
- Do not write service/current-student facts into a new-sales deal unless that deal is explicitly about renewal/upsell.
- Do not overwrite non-empty deal AI fields in live mode until readback and audit gates are green.

## Recommended rollout

1. Build read-only unified client timeline: calls + AMO + Tallanto.
2. Build deal matching dry-run: phone/contact -> AMO deals -> Tallanto product/payment/group signals -> selected deal or manual review.
3. Generate a preview workbook with contact summary and deal summary side by side.
4. ROP reviews 20-50 rows.
5. Create missing AMO deal fields if needed.
6. Run dry-run writeback to deals.
7. Claude/GPT audit.
8. Staged live writeback to deals: 20 -> 100 -> 300 -> larger batches, with readback after every stage.
