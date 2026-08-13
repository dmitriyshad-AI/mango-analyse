# QUALITY_REPORT kb_release_20260813_v6_8_owner_approved

quality_passed: `True`

## Summary
- facts_total: `806`
- client_allowed_facts: `471`
- source_registry_total: `13`
- approval_queue_items: `784`
- approval_queue_by_type: `{'price': 28, 'discount': 11, 'promocode': 13, 'installment': 14, 'deadline': 7, 'camp_lvsh': 6, 'camp_zvsh': 2, 'program': 255, 'matkap': 24, 'documents': 67, 'location': 1, 'teacher': 13, 'refund': 18, 'policy': 19, 'schedule': 178, 'process': 61, 'contacts': 3, 'format': 6, 'intensive': 34, 'tax': 10, 'contact': 14}`

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
- found: `31`
- missing: `[]`

## Blocking Failures
- none

## Stage 6
- status: `not_run_by_builder`
- note: Сборщик готовит v3 snapshot и fixtures-compatible поля; Stage 6 запускается отдельным безопасным тестовым контуром.
