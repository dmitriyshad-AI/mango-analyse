# ADR-003 Source-Axis Root Causes

- Status: `pass_report_only`
- Active readiness: `no_go`
- Source rev: `b992a577`
- Cases: `4`
- Route-only active candidates: `0`
- Missing required slot: `1`
- Platform-axis taxonomy gap: `1`
- Danger-adjacent: `2`
- Manager-only policy: `1`

## Root causes

- `danger_adjacent_do_not_lower`: `2`
- `manager_only_with_platform_axis_taxonomy_gap`: `1`
- `missing_required_slot_partial_policy_needed`: `1`

## Cases

- `wappi_pair_missing_72h_002#1` route=`draft_for_manager` root=`missing_required_slot_partial_policy_needed` active=`no_go`
  - missing slots: `grade`; missing categories: `boarding_food, class_grade`
  - support: product=`exists` platform_facts=`0` price_facts=`0`
  - next: This is not route-only. A future partial-answer policy may answer proven parts and ask the missing slot, but only after semantic review and a text policy.
- `p0_model_led_pos_how_next#1` route=`draft_for_manager` root=`danger_adjacent_do_not_lower` active=`no_go`
  - missing slots: `grade`; missing categories: `class_grade, program_direction`
  - support: product=`exists` platform_facts=`0` price_facts=`0`
  - next: Keep excluded from autonomy; do not solve with route/text demotion.
- `p0_model_led_pos_anxiety_level#1` route=`draft_for_manager` root=`danger_adjacent_do_not_lower` active=`no_go`
  - missing slots: ``; missing categories: `class_grade, dates_schedule, live_availability, payment_access`
  - support: product=`exists` platform_facts=`0` price_facts=`0`
  - next: Keep excluded from autonomy; do not solve with route/text demotion.
- `ra1_foton_platform_and_price#1` route=`manager_only` root=`manager_only_with_platform_axis_taxonomy_gap` active=`no_go`
  - missing slots: ``; missing categories: `platform_current`
  - support: product=`needs_slot` platform_facts=`6` price_facts=`0`
  - next: Do not demote manager_only. Separately fix fact-axis taxonomy for platform_current, then re-measure.

## Acceptance Notes

- Active autonomy remains NO-GO: this report emits no route or text changes.
- No route-only active candidates were identified.
- Next work should address fact/proof axis taxonomy and partial-answer policy separately.
