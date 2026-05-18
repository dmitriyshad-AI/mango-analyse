# QUALITY_REPORT kb_release_20260518_v3

quality_passed: `True`

## Summary
- facts_total: `664`
- client_allowed_facts: `360`
- source_registry_total: `10`
- approval_queue_items: `641`
- approval_queue_by_type: `{'price': 70, 'discount': 55, 'promocode': 27, 'installment': 18, 'matkap': 36, 'deadline': 26, 'camp_zvsh': 2, 'program': 195, 'documents': 60, 'teacher': 11, 'refund': 18, 'policy': 4, 'camp_lvsh': 52, 'camp_city': 13, 'intensive': 38, 'tax': 5, 'contact': 11}`

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
- found: `25`
- missing: `[]`

## Blocking Failures
- none

## Stage 6
- status: `not_run_by_builder`
- note: Сборщик готовит v3 snapshot и fixtures-compatible поля; Stage 6 запускается отдельным безопасным тестовым контуром.
