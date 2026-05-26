# RUNBOOK: сборка релиза базы знаний (KB). 2026-05-26

Готов к размещению в `docs/KB_BUILD_RUNBOOK_2026-05-26.md` (применяет Codex/Дмитрий).
Цель — чтобы базу всегда собирали ПРАВИЛЬНЫМ билдером и не повторили тихую порчу фактов.

## Каноничный релиз
`product_data/knowledge_base/kb_release_20260520_v6_3_team_answers/`
Снимок для бота/симулятора: `.../kb_release_v3_snapshot.json` (схема v3; релиз v6.3; билдер v6.1).
v3.3 (`kb_release_20260518_v3_3*`) — УСТАРЕЛА, не источник правды.

## Команда сборки (ТОЛЬКО эта)
```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/build_kb_release_v6_1_team_answers.py
```
Билдер читает `builder_version` и `release_manifest.yaml` из
`kb_release_20260520_v6_3_team_answers_sources/` и применяет дельту контрольных чисел (remove/add).

## ЧТО НЕЛЬЗЯ
НЕ запускать `scripts/build_kb_release_v3_from_claude_handoff.py`: у него старый `BUILDER_VERSION`,
дефолтный out-dir `v4`, и он НЕ применяет v6.3-манифест → ложный `quality_passed=false` и риск порчи
текста факта (был реальный случай «занятия проходят 2 026 раз в неделю» — год утёк в числовое поле).

## Обязательная пред-проверка (gate) после сборки
1. `quality_report.json`: `quality_passed=true` и зелёные чек-поля:
   `text_number_grounded`, `field_ranges_ok`, `weekly_frequency_is_plausible`, `control_numbers_present`.
2. `semantic_review.json`: `semantic_pass=true`, blocking 0.
3. `control_numbers_missing=[]`, `source_registry_total=11`.
4. **Целостность текста (важно — гейт контрольных чисел этот класс НЕ ловит):** прогнать проверку
   «число в client_safe_text сверяется со structured_value/raw_value того же факта» + sanity-диапазоны
   полей (weekly_frequency 1..7, weeks 1..52, percentage 0..100, classes 1..11, деньги >0). На стороне
   Claude есть независимый чекер `wave1_acceptance_integrity_check.py` для сверки.
5. Грепнуть client_safe_text ИЗМЕНЁННЫХ фактов на абсурд: `N NNN раз`, число против структуры.
6. Бренд-чистота: client-safe текст не содержит чужой бренд (Фотон в УНПК и наоборот).

## Приёмка
Релиз считается готовым, только если §1-§6 зелёные. Зелёные unit-тесты сами по себе = formal_pass,
не semantic_pass — не путать. Спорные числовые/смысловые факты — регрейдить вручную по client_safe_facts.

## Имена версий (чтобы не путать)
- релиз: v6.3 · схема снимка: v3 (`kb_release_v3_snapshot.json`) · билдер: v6.1.
Три разных «v»-номера на одном артефакте — это нормально, но источник прошлой путаницы.
