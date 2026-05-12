# Stage 15 Claude Safety Fix Report - 2026-05-10

## Что исправляли

Claude Stage 15 audit нашел P0-утечки конкретных цен в `bot_safe_answer`: `7900`, `88000`, `147000`, `78400`. Также были P1/P2 риски: адреса, имена педагогов, относительные дедлайны, soft promises, платежные провайдеры и broken email fragments.

Корень проблемы: старый sanitizer хорошо ловил часть ценовых формулировок, но не покрывал naked 5-6 digit amounts в конструкциях вроде `за 88000`, `год целиком за 147000`, `при оплате 78400`. Кроме того, Stage 15 gate раньше проверял не весь экспортный ряд достаточно жестко и не делал повторную sanitizer-нормализацию на финальном export layer.

После первичной пересборки дополнительный literal-scan выявил остаточный P1-адресный паттерн `Скорняжный/Скорняжном` и `Чистыми прудами`. Он также закрыт и downstream пересобран как `v7_location_fix`.

## Реализованные шаги

1. Заморожен предыдущий allowlist как `PRE_FIX_DO_NOT_USE_FOR_PRODUCTION_BOT`, чтобы его нельзя было случайно использовать для production-бота.
2. Усилен `src/mango_mvp/insights/sanitizers.py`:
   - цены и крупные naked money amounts;
   - broken email fragments;
   - платежные провайдеры;
   - имена педагогов после ролевых слов;
   - адреса/метро/кабинеты, включая `Скорняжный/Скорняжном` и `Чистые/Чистыми пруды`;
   - ссылки на договоры/оферты/документы;
   - soft promises и относительные дедлайны.
3. Усилен `src/mango_mvp/quality/stage15_export_quality_gate.py`:
   - финальный export row повторно проходит sanitizer;
   - проверяются не только ответы, но и вопрос/ограничения/when-not-use;
   - unsafe source rows блокируются до попадания в allowlist;
   - добавлены source/export risk counts.
4. Добавлены регрессионные тесты на P0/P1/P2 паттерны из аудита Claude.
5. Пересобраны KB, ROP, baseline, Stage 14 и Stage 15 артефакты на исправленном sanitizer/export gate.
6. Собран отдельный пакет для повторного Claude/GPT-аудита.

## Финальный статус Stage 15

- Актуальный root: `stable_runtime/transcript_quality_stage15_export_gate_20260510_v7_location_fix_row_scan/`
- `passed`: true
- `rop_internal_export_ready`: true
- `crm_quality_writeback_ready`: true
- `bot_allowlist_export_ready`: true
- `bot_autonomous_production_ready`: false
- Причина блокировки autonomous bot: остается очередь `over_sanitization_queue`, которую надо проверить/переписать до автономного бота.

## Ключевые метрики финального allowlist

- `bot_export_allowlist_rows`: 472
- `blocked_bot_export_rows`: 0
- `stage14_residual_risk_rows`: 0
- `stage14_over_sanitization_rows`: 250
- Все финальные `risk_counts`: 0
- Все `source_risk_counts` по KB и ROP bot drafts: 0

Важно: не все P0-строки из аудита обязаны исчезнуть из allowlist как строки. Часть может остаться, если финальный текст безопасно переписан и в нем больше нет конкретных сумм/адресов/ПДн/брендовых утечек. В финальном Stage 15 unsafe literals отсутствуют, а gate показывает нулевые риски.

## Проверки

- Narrow regression tests: 4 passed.
- Expanded knowledge-base/stage14/stage15/ROP tests: 25 passed.
- Full suite: 690 passed, 82 warnings.
- Literal grep по финальному allowlist не нашел P0/P1/P2 паттерны: `7900`, `88000`, `147000`, `78400`, адреса, имена педагогов, `до конца дня/года/каникул`, `Альфа`, email/phone patterns.

## Остаточные ограничения

- Autonomous bot пока не включать: надо разобрать `over_sanitization_candidates.csv`, чтобы не потерять полезность ответов.
- CRM/ROP internal export готов, но live writeback все равно должен идти через staged dry-run/live guard.
- Для коммерческого SaaS нужно вынести brand/address/teacher/payment-provider словари в tenant YAML/DB config, а не держать только regex в коде.

## Основные файлы

- `stable_runtime/transcript_quality_stage15_export_gate_20260510_v7_location_fix_row_scan/summary.json`
- `stable_runtime/transcript_quality_stage15_export_gate_20260510_v7_location_fix_row_scan/STAGE15_EXPORT_GATE_REPORT.md`
- `stable_runtime/transcript_quality_stage15_export_gate_20260510_v7_location_fix_row_scan/bot_export_allowlist.csv`
- `stable_runtime/claude_stage15_safety_fix_reaudit_package_20260510_v2_location_fix/`
