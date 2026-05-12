# Tallanto context and abonement logic, 2026-05-12

This file captures the current working understanding of Tallanto data that should feed Mango Analyse CRM/deal-aware enrichment.

Tallanto remains read-only for Mango Analyse in the current production policy. AI can read and summarize Tallanto context into AMO AI fields, but must not write back to Tallanto.

## Source artifacts

- User-provided write-off report: `/Users/dmitrijfabarisov/Projects/Mango analyse/260512_190605_117075d7_write_off_visits_from_class.xlsx`
- Normalized report CSV: `/Users/dmitrijfabarisov/Projects/Mango analyse/stable_runtime/tallanto_write_off_visits_20260512/write_off_visits_from_class_normalized.csv`
- Report summary: `/Users/dmitrijfabarisov/Projects/Mango analyse/stable_runtime/tallanto_write_off_visits_20260512/summary.json`
- Historical write-off combined unique CSV: `/Users/dmitrijfabarisov/Projects/Mango analyse/stable_runtime/tallanto_write_off_visits_history_20260512/write_off_visits_combined_unique.csv`
- Historical write-off by-student aggregate: `/Users/dmitrijfabarisov/Projects/Mango analyse/stable_runtime/tallanto_write_off_visits_history_20260512/write_off_visits_by_student.csv`
- Historical write-off summary: `/Users/dmitrijfabarisov/Projects/Mango analyse/stable_runtime/tallanto_write_off_visits_history_20260512/summary.json`
- Extended Tallanto schema export: `/Users/dmitrijfabarisov/Projects/Mango analyse/stable_runtime/tallanto_schema_extended_20260512/tallanto_fields_extended.json`

## Write-off report findings

The report `Списания за посещение занятий` covers `2026-01-01` to `2026-04-30`.

The XLSX file has an incorrect Excel dimension marker (`A1`), so standard Excel readers may see it as an empty sheet. The actual data exists in the sheet XML.

Parsed structure:

- rows: `30,969`
- unique barcodes: `1,608`
- unique student names: `1,607`
- total write-off sum: `61,167,175.41`
- average write-off per row: `1,975.11`

Columns:

- `Фамилия`
- `Имя`
- `Штрихкод`
- `Абонемент`
- `Сумма списания`
- `Дата списания`
- `Тип списания`
- `Занятие`
- `Филиал занятия`
- `Дата занятия`
- `День рождения`

Observed values:

- `Абонемент`: mostly `Абонемент`, with some `Разовое`
- `Тип списания`: payment/source-like values such as `Webmoney`, `Альфа-банк`, `Единая касса`, `СБП`, `Безналичный расчет`, `Кредит`
- `Занятие`: contains compact class metadata: subject, year, format, grade, schedule, teacher, room/group hints
- `Филиал занятия`: `Онлайн`, `Сретенка`, `МФТИ`, `ШД`, `Онлайн АНО`, `Пацаева`, etc.

Interpretation:

This report is the strongest current source for actual attendance-backed revenue consumption. It proves that the student did not just pay, but that money was written off against a specific attended/scheduled class.

## Historical write-off reports, 2024-06 through 2026-04

Additional reports were added on 2026-05-12:

| File | Reported period | Parsed rows | Notes |
|---|---:|---:|---|
| `260512_193755_e8470914_write_off_visits_from_class.xlsx` | `2025-09-01` to `2025-12-31` | `33,718` | Fully covered by the larger 2024-2025 report; useful as validation |
| `260512_193833_b7a7653b_write_off_visits_from_class.xlsx` | `2025-01-01` to `2025-08-31` | `52,739` | Fully covered by the larger 2024-2025 report; useful as validation |
| `260512_194037_a0c80088_write_off_visits_from_class.xlsx` | `2024-06-01` to `2025-12-31` | `140,924` | Main historical report for 2024-2025 |
| `260512_190605_117075d7_write_off_visits_from_class.xlsx` | `2026-01-01` to `2026-04-30` | `30,969` | Main 2026 report currently available |

Combined historical result:

- input rows including overlaps: `258,350`
- exact duplicate rows removed: `86,458`
- unique write-off rows: `171,892`
- actual coverage: `2024-06-01 10:00` to `2026-04-30 18:15`
- unique student/barcode aggregate rows: `5,029`
- unique student-name count in row-level data: `5,024`
- total write-off amount across unique rows: `390,591,743.52`
- average write-off row amount: `2,272.31`

Important overlap conclusion:

The two smaller 2025 files are not additive. They are contained in the large `2024-06-01` to `2025-12-31` report. For production history restoration, use:

1. large `2024-06-01` to `2025-12-31` report;
2. `2026-01-01` to `2026-04-30` report;
3. smaller 2025 reports only as validation/cross-check.

This gives a clean historical attendance/write-off layer from June 2024 through April 2026.

Month-level unique row counts:

| Month | Rows |
|---|---:|
| 2024-06 | 694 |
| 2024-07 | 1,719 |
| 2024-08 | 608 |
| 2024-09 | 7,858 |
| 2024-10 | 14,013 |
| 2024-11 | 14,781 |
| 2024-12 | 14,793 |
| 2025-01 | 10,055 |
| 2025-02 | 11,482 |
| 2025-03 | 11,897 |
| 2025-04 | 9,557 |
| 2025-05 | 9,179 |
| 2025-06 | 215 |
| 2025-07 | 227 |
| 2025-08 | 127 |
| 2025-09 | 6,010 |
| 2025-10 | 9,049 |
| 2025-11 | 10,448 |
| 2025-12 | 8,211 |
| 2026-01 | 7,124 |
| 2026-02 | 7,963 |
| 2026-03 | 8,451 |
| 2026-04 | 7,431 |

Branch-level unique row leaders:

- `Онлайн`: `67,231`
- `Сретенка`: `39,153`
- `МФТИ`: `30,381`
- `ШД`: `17,051`
- `Пацаева`: `9,418`
- `Онлайн АНО`: `6,975`
- `Выездные школы (иногородний)`: `1,024`

Write-off type leaders:

- `Webmoney`: `88,284`
- `Единая касса`: `31,242`
- `Альфа-банк`: `29,013`
- `СБП`: `10,381`
- `Безналичный расчет`: `6,785`
- `ЦРДО РС`: `2,023`
- `Т-Банк`: `1,990`

Interpretation for Mango Analyse:

The historical reports are valuable for backfilling attendance/write-off facts. They should not become the long-term product integration method. For SaaS/product operation, Mango Analyse should fetch this information from Tallanto API or a scheduled Tallanto export, then compare it against report-based snapshots during validation.

## Tallanto modules that matter

Confirmed modules from live schema:

| Business area | Tallanto module | Why it matters |
|---|---|---|
| Student/contact | `Contact` | Parent/student identity, phones, branch, type, AMO ID, communication history, interests |
| Tallanto deal | `Opportunity` | Historic Tallanto sales context; often imperfect, but useful for interest/outcome |
| Request/application | `Request` | Incoming requests, next contact date, source/status |
| Finance | `most_finances` | Payments, write-offs, refunds, checks, invoice/class/abonement linkage |
| Abonement | `most_abonements` | Purchased package: dates, cost, visits/hours, remaining balance, freezing, discount |
| Abonement template | `most_template_abonements` | Product/template behind abonement: duration, cost, visit count, accounting method |
| Group | `most_courses` | Group/course membership context |
| Class/lesson | `most_class` | Specific lesson: subject, date, teacher, branch, group, cost |
| Student-group relation | `CoursesContactsRelationship` | Whether student is/was recorded into a group |
| Student-class relation | `ClassContactsRelationship` | Whether student is/was recorded into a lesson; status, abonement used, duration, rating |

## Abonement logic: current understanding

An abonement in Tallanto is a paid package attached to a student.

Key fields in `most_abonements`:

- `contact_id`: student who paid/owns the abonement
- `template_id`: link to abonement template
- `type`, `form`, `category`, `rate`: package classification
- `start_date`, `finish_date`: validity period
- `cost`: total package cost
- `num_visit`: total included visits/hours
- `num_visit_left`: remaining visits/hours
- `num_visit_type`: accounting method, likely visits vs hours
- `class_cost_for_inclusive`: revenue attributed to one class visit under this package
- `freezing_count`, `max_freezing_days`: freezing logic
- `max_number_absent`: allowed absences
- `discount`, `discount_comment`: discount context
- `invoice_id`: source invoice
- `filial`: branch

Key fields in `most_template_abonements`:

- `cost`: template package price
- `num_visit`: included visits/hours
- `num_visit_type`: accounting method
- `duration`: validity duration
- `class_cost_for_inclusive`: planned per-class revenue
- `write_contact_from_abonement`: whether package can auto-record student
- `calendar_type`, `max_freezing_days`, `max_number_absent`

Key fields in `most_finances`:

- `direction`: payment direction, expected to separate incoming/outgoing/write-off/refund flows
- `cost`: amount
- `date_payment`: operation date
- `contact_id`: student
- `most_abonements_id`: linked abonement
- `most_class_id`: linked class/lesson, if operation is class-specific
- `invoice_id`: linked invoice
- `type`: payment method
- `print_check_status`, `print_refund_status`: fiscal receipt status
- `tags`: operation labels

Key fields in `ClassContactsRelationship`:

- `contact_id`: student
- `most_class_id`: class/lesson
- `most_class_contacts_status`: enrollment/visit status
- `most_class_abonements`: abonement used for this class
- `most_class_contacts_duration`: lesson duration in hours
- `most_class_contacts_cost`: cost option
- `most_class_discount`, `most_class_discount_type`: class-level discount
- `date_entry`: enrollment date
- `amo_id`: AMO ID linkage if present

## Practical business meaning

For CRM recommendations, Tallanto must answer five different questions:

1. Is the client just a lead, or already a real student?
2. What product/group/subject has the student actually attended?
3. Is there real payment, only invoice/intent, or actual write-off for visits?
4. Is there remaining balance/active package, or has the package ended?
5. Does the current AMO deal match the actual Tallanto product/group/payment context?

Payment alone is not enough.

Examples:

- Payment exists, but no write-offs: likely paid but lessons have not started, or package not consumed yet.
- Write-offs exist: student actually attended/scheduled lessons and revenue was consumed.
- Active group/class relation exists: student is operationally enrolled.
- `num_visit_left` is low/zero: possible renewal/upsell signal.
- Several groups/abonements exist: contact-level summary must not be copied blindly into one deal.

## How this should change Mango Analyse

### Contact-level AI context

Contact summary should include a compact Tallanto block:

- student identity/class/type;
- active/recent groups;
- recent subjects;
- active abonement/payment status;
- recent write-off/attendance signal;
- unresolved duplicate/matching risks.

### Deal-level AI context

Deal summary should include only Tallanto facts relevant to that deal:

- matching product/group/subject;
- payment/abonement/write-off status for this product;
- whether client already paid/attended;
- whether current deal is probably stale, duplicate, closed, or next-year perspective.

### Quality gates

Before writing AI recommendations to AMO:

- If Tallanto shows payment/write-off for the same product, AI must not recommend “send payment link” unless the deal context explicitly says second payment is needed.
- If Tallanto shows active attendance, AI must treat the client as existing student/service/renewal context, not as a cold lead.
- If multiple active Tallanto groups/products exist, deal writeback must require high-confidence deal-product matching.
- If abonement is expired/empty but product interest continues, AI should suggest renewal/follow-up, not first-sale messaging.
- If payment exists but write-off does not, AI should distinguish “paid, awaiting start/enrollment” from “unpaid”.

## Gaps to clarify with operations

These points need either Tallanto API examples or discussion with the team:

- Exact values of `most_finances.direction` and how to identify incoming payment vs write-off vs refund.
- Whether `Списания за посещение занятий` is generated from `most_finances`, `ClassContactsRelationship`, or an internal Tallanto report layer.
- How `num_visit_left` changes after absence, freezing, manual correction, refund, or class cancellation.
- Whether `Разовое` in the report always means a one-off payment rather than abonement consumption.
- Whether `Тип списания` is strictly payment method or a broader financial operation source.
- Which status values in `most_class_contacts_status` mean attended, planned, missed, canceled, or reserved.

Until these are resolved, Mango Analyse should use Tallanto finance/abonement data as a strong signal, but with conservative wording and fail-closed gates for risky recommendations.
