# Stage 15 Re-Audit Fix Report - 2026-05-10

## Что было найдено Claude во втором аудите

Claude подтвердил, что P0-цены, прежние адреса, платежные провайдеры и `до конца X` были устранены. Остались новые классы:

- P1: orphan-фамилии/отчества после частичной замены ФИО (`ученик Гамзяков`, `ученик Еделькина`, `Кондрашова`, `Камаринцев`, `Николаевне`).
- P2: даты вида `до 15 числа`, `4 и 11 числа`.
- P2: `компенсировать` как soft-promise без temporal-якоря.
- P1/P3: `Майская`, `КПМ`, `кабинетом 324`.
- Отдельный customer-question leak: `Екатерину` в sanitized question example.

## Что исправлено

В `src/mango_mvp/insights/sanitizers.py` добавлены и подключены:

- `ORPHAN_NAME_AFTER_PLACEHOLDER_RE`
- `FAMILY_NAME_LABEL_RE`
- `ACTION_VERB_SURNAME_RE`
- `CONTEXT_SURNAME_RE`
- расширение `TEACHER_NAME_RE` до 4 слов после role marker
- `COMPENSATION_LANG_RE`
- расширение `DEADLINE_RE` на `N числа` и `N и M числа`
- расширение `ADDRESS_RE` на `Майская`, `КПМ`, inflected `кабинетом 324`
- расширение `COMMON_SINGLE_NAME_RE` на `Катерина/Екатерина` в нужных падежах

В тесты добавлены регрессии:

- orphan surnames/patronymics after `ученик`
- `фамилию Николаев`
- `будет вести Кондрашова`
- `Камаринцев вести`
- `Гамзяков/Еделькина`
- `до 15 числа`, `до 17 числа`, `4 и 11 числа`
- `компенсировать`
- `КПМ/Майская/кабинет 324`

## Пересборка

Пересобраны downstream-артефакты:

- KB: `stable_runtime/sales_insight_knowledge_base_after_quality_backfill_20260510_v8_orphan_surname_fix/`
- ROP: `stable_runtime/rop_validation_pack_after_quality_backfill_20260510_v8_orphan_surname_fix/`
- Baseline: `stable_runtime/transcript_quality_baseline_after_quality_backfill_20260510_v8_orphan_surname_fix/`
- Stage 14: `stable_runtime/transcript_quality_stage14_comparison_20260510_v6_orphan_surname_fix/`
- Final Stage 15: `stable_runtime/transcript_quality_stage15_export_gate_20260510_v9_customer_name_fix/`

## Финальный Stage 15

- `passed`: true
- `bot_export_allowlist_rows`: 471
- `blocked_bot_export_rows`: 0
- `stage14_residual_risk_rows`: 0
- `stage14_over_sanitization_rows`: 250
- все `bot_export.risk_counts`: 0
- все `source_risk_counts`: 0
- `bot_allowlist_export_ready`: true
- `bot_autonomous_production_ready`: false, потому что остается over-sanitization queue

## Контрольный adversarial scan

На финальном `bot_export_allowlist.csv` проверены группы:

- P0 prices: `7900`, `88000`, `147000`, `78400`
- Claude P1 names/addresses: `Гамзяков`, `Кондрашова`, `Еделькина`, `Камаринцев`, `Николаев`, `Алексеевич`, `Александровна`, `Иванович`, `Катерина/Екатерина`, `Майская`, `КПМ`, `кабинет 324`
- Claude P2: `до N числа`, `N и M числа`, `компенсировать`, `до конца дня/года/каникул`
- contacts: email/phone-like patterns

Результат: 0 hits по всем этим группам.

## Тесты

- Targeted regression: 2 passed.
- Related tests: 27 passed.
- Full suite: 692 passed, 82 warnings.

## Остаточные ограничения

- Autonomous bot пока не включать: `over_sanitization_candidates.csv` все еще требует проверки полезности/переписывания.
- В controlled allowlist могут оставаться P3 tenant-context слова уровня `Москва/ФТИ`; Claude ранее счел их допустимыми для одного tenant, но для SaaS нужно вынести brand/address/city dictionaries в tenant config.
