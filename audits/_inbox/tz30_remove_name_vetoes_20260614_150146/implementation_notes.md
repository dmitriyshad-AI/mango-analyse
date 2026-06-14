# TZ30 remove name vetoes

Дата: 2026-06-14.

Сделано:

- `PROMPT_VERSION` поднят с `v3` до `v4`.
- Убраны детерминированные name-вито:
  - `mention_name_misattributed`;
  - `different_child_names_merged`;
  - `child_names_are_compatible`;
  - `_NAME_ALIASES`;
  - Levenshtein identity check.
- Оставлены механические проверки:
  - `invented_child_name`;
  - empty/unknown/duplicate/incomplete mention mapping;
  - count-cap `children_count_exceeds_*`;
  - physical grade guard `incompatible_grades_same_period`.
- `merge_confidence` остаётся логом: schema требует поле для Codex structured output, но локальная нормализация missing/invalid значения дефолтит в `low` и не отклоняет семью.
- `child_key` теперь строится из hash отсортированного множества `mention_id` группы, а не из `canonical_name`.
- Raw name diagnostics сохранены как local ignored review artifact: `name_review_diagnostics.local.jsonl`.
- Скрипт микропробы добавляет anonymized `low_confidence_multi_named.anonymized.jsonl` для ручного регрейда.

Микропроба v4:

- out-root: `/Users/dmitrijfabarisov/Projects/Mango_tz24_dedup/product_data/customer_profiles/tz30_microprobe_v4_20260614_142934`
- selected_count: 120
- trace_events: 120
- accepted: 108
- fail-soft: 11
- shared-phone skipped: 1
- errors: `incompatible_grades_same_period=5`, `incomplete_mention_mapping=4`, `unknown_mention_id=1`, `invented_child_name=1`
- name-veto errors: 0
- raw name review diagnostics: 79 rows, local-only
- low-confidence multi-named queue: 48 rows
- known-bad found: 4/4

Неудачный diagnostic run:

- `/Users/dmitrijfabarisov/Projects/Mango_tz24_dedup/product_data/customer_profiles/tz30_microprobe_v4_20260614_140236`
- Причина: optional schema property не принял Codex structured output; исправлено возвратом `merge_confidence` в schema `required` без enum.
- Этот run ignored и не используется как acceptance-result.
