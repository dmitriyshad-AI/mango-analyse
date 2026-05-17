# QUALITY_REPORT kb_release_20260518_v3

quality_passed: `True`

## Summary
- facts_total: `726`
- client_allowed_facts: `419`
- source_registry_total: `10`
- approval_queue_items: `705`
- approval_queue_by_type: `{'price': 124, 'discount': 65, 'promocode': 11, 'installment': 18, 'tax': 30, 'matkap': 40, 'refund': 2, 'deadline': 26, 'camp_lvsh': 64, 'camp_zvsh': 2, 'program': 172, 'intensive': 42, 'documents': 66, 'contact': 13, 'location': 2, 'teacher': 11, 'policy': 4, 'camp_city': 13}`

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
