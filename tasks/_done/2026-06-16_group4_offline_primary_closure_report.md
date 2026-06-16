# Group 4 Offline Modes Closure Report

Дата: 2026-06-16  
Ветка текущего отчета: `codex/tz125-finalize-group4`

## Итоговое решение

Группа 4 закрыта в режиме офлайн-аналитики:

| Блок | Статус | Решение |
|---|---|---|
| B — исход сделки | `primary` разрешён | Только allowlist flip `won_paid_or_active -> known_student_or_lead`; `payment_pending` не применять |
| D — роли моно-звонков | `primary(clean)` разрешён | Источник `b00b3bd`; `codex_selective` только на low-confidence; уверенное правило остается правилом; segment guard не входит в clean-дифф |
| E — бренд | `primary` разрешён | `cyrillic_v2`; Foton по корню, cross-brand/unknown остаются fail-closed |
| C — категории вопросов | `primary` разрешён | Гибрид: модель по умолчанию, guard для разобранных регрессий и реально уверенных служебных случаев |
| A — качество сделки | `shadow` оставить | Primary не включать: `23/24` модель против `22/24` правило = шум + `1` уверенная ошибка модели |

## Что важно

- Это не live-write и не запись в AMO/Tallanto/CRM.
- Default-off инвариант сохраняется: новые primary-режимы включаются только явным офлайн-режимом/флагом.
- A остается диагностическим shadow-контуром и не влияет на решения.
- Основной каталог вопросов не пересобирался.
- ASR не запускался.
- OpenAI API key не используется.

## Источники решения

- B: `tasks/_done/2026-06-16_TZ121_block_B_primary_enabled_report.md`
- E: `tasks/_done/2026-06-16_TZ121_block_E_primary_enable_report.md`
- C: `tasks/_done/2026-06-16_TZ121_block_C_primary_enable_report.md`
- A: `tasks/_done/2026-06-16_TZ121_block_A_deal_gold_shadow_regrede_stop_report.md`
- D: clean commit `b00b3bd` на ветке `codex/tz118-d-primary-clean`, отчет `tasks/_done/2026-06-16_TZ118_block_D_primary_clean_report.md`

## Сводка метрик

- B: разрешён только подтвержденный flip `won_paid_or_active -> known_student_or_lead`; `payment_pending` оставлен legacy.
- E: follow-up срез `20` строк; `foton->unknown` исправлено `9/9`, expected fail-closed осталось `8/8`.
- C: гибрид `80/100`, модель `72/100`, правило `37/100`; primary включён только для офлайн-классификации.
- D: clean `codex_selective` сохраняет принятую метрику `94.62%` средней точности по репликам против около `55%` у слабого правила; `segment_guard` удалён из clean-источника и не меняет принятую метрику, потому что отклонённый слой был force-off/не применялся.
- A: модель `23/24`, правило `22/24`, high-confidence wrong model rows `1`; primary отклонен.

## Semantic Review

Verdict: `PASS_WITH_NOTES`.

Что прошло:

- Решение по A не переоценивает слабый прирост модели.
- Блоки B/D/E/C ограничены офлайн-режимами и не дают записи во внешние системы.
- Брендовый fail-closed в E не ослаблен.
- D не возвращает отклоненный segment guard.

Остаточные риски:

- A требует большего gold-набора, если позже понадобится пересмотреть shadow-only решение.
- D имеет известный остаточный риск модельных сегментных путаниц около `6%`, чинить его нужно отдельным модельным улучшением, не правиловым guard.

## Проверки

Текущая ветка B/E/C/A:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q

3297 passed, 5 skipped, 1 warning
```

D clean-ветка `codex/tz118-d-primary-clean` по commit `b00b3bd`, перенесённая в `codex/tz125-finalize-group4`:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q

3311 passed, 5 skipped, 1 warning in 54.21s
```

Регрессия `test_tz121_*` на ветке `codex/tz125-finalize-group4`:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_tz121_*.py

9 passed in 0.84s
```

## Следующий возможный шаг

Только если потребуется возвращаться к A: добрать расширенный gold-набор по сделкам, отдельно измерить модель против правила и заново вынести решение. До этого A остается `shadow`.
