# QUALITY_REPORT kb_release_20260518_v3

quality_passed: `True`

## Summary
- facts_total: `1045`
- client_allowed_facts: `668`
- source_registry_total: `11`
- approval_queue_items: `1023`
- approval_queue_by_type: `{'price': 49, 'discount': 67, 'promocode': 13, 'installment': 29, 'tax': 40, 'matkap': 50, 'deadline': 133, 'camp_lvsh': 120, 'camp_zvsh': 3, 'program': 290, 'documents': 67, 'teacher': 13, 'refund': 18, 'policy': 10, 'process': 54, 'contacts': 2, 'camp_city': 17, 'intensive': 34, 'contact': 14}`

## Checks
- all_fact_source_ids_exist: `True`
- all_claude_sources_have_sha256: `True`
- control_numbers_present: `True`
- no_empty_fact_text: `True`
- forbidden_to_say_not_in_facts: `True`
- allowed_client_text_has_no_license_numbers: `True`
- weekly_frequency_is_plausible: `True`
- text_number_grounded: `True`
- field_ranges_ok: `True`
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
