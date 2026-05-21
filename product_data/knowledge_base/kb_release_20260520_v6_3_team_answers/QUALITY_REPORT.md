# QUALITY_REPORT kb_release_20260518_v3

quality_passed: `True`

## Summary
- facts_total: `720`
- client_allowed_facts: `397`
- source_registry_total: `10`
- approval_queue_items: `702`
- approval_queue_by_type: `{'price': 46, 'discount': 59, 'promocode': 13, 'installment': 18, 'tax': 38, 'matkap': 42, 'deadline': 24, 'camp_zvsh': 2, 'program': 210, 'documents': 62, 'teacher': 13, 'refund': 18, 'policy': 4, 'camp_lvsh': 90, 'camp_city': 17, 'intensive': 34, 'contact': 12}`

## Checks
- all_fact_source_ids_exist: `True`
- all_claude_sources_have_sha256: `True`
- control_numbers_present: `True`
- no_empty_fact_text: `True`
- forbidden_to_say_not_in_facts: `True`
- allowed_client_text_has_no_license_numbers: `True`
- allowed_client_text_passes_brand_safety: `True`
- approval_queue_has_required_columns: `True`
- approval_queue_has_400_plus_items: `True`
- approval_queue_has_business_types: `True`
- brand_scope_has_foton_and_unpk_facts: `True`
- post_filter_has_phrases: `True`
- two_separate_bots_recorded: `True`

## Control Numbers
- found: `20`
- missing: `[]`

## Blocking Failures
- none

## Stage 6
- status: `not_run_by_builder`
- note: Сборщик готовит v3 snapshot и fixtures-compatible поля; Stage 6 запускается отдельным безопасным тестовым контуром.
