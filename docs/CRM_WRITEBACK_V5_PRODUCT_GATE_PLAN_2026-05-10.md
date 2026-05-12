# CRM Writeback v5 Product Gate Plan

Дата: 2026-05-10
Статус: реализация v5 после Claude v4 и class-discovery аудита.

## Цель

Перестать закрывать частные строки из аудита и перевести AMO-ready слой в продуктовый режим: правила должны работать для будущего SaaS, других клиентов и новых формулировок звонков.

## Классы, которые закрывает v5

1. A4 `service/existing-client as new lead`.
   - Не считаем `service_call`, `existing_client_progress`, `technical_call` live-ready sales writeback.
   - Такие строки остаются в master contacts/manual review, но не уходят в AMO-ready как новые лиды.

2. B6/C9 `orphan/no exact AMO entity`.
   - Для live-ready требуется ровно один `AMO contact ID`.
   - Пустой или множественный AMO contact переводится в manual/orphan queue.
   - Live writeback имеет defensive guard, даже если CSV собран старым скриптом.

3. C1/F5 `population-level recall`.
   - Добавлен независимый population marker counter рядом с detector.
   - Gate теперь различает `detector passed` и `population markers still present`.

4. C8/F8 `corpus self-validation loop`.
   - Frozen corpus расширен не только прошлыми Claude findings, но и forward/synthetic/random/negative-overblock layers.
   - Summary содержит seed-policy и rolling-closure статус; класс нельзя объявлять закрытым только по одному аудиту.

5. A2/A3 `out-of-domain/no-content long tail`.
   - Detector расширен обобщенными грамматиками: wrong site/resource, carrier/corporate with context, partnership pitch, no-dialogue/no-response, missing intent.
   - Добавлены anti-overblock cases: `МТС Линк`, валидный отказ из-за нехватки времени, existing-client service.

6. C12 `history duplication`.
   - Контактная сводка больше не встраивает полный latest summary.
   - Хронология стала компактнее.
   - Writeback не добавляет хронологию, если она почти полностью дублирует сводку.
   - Quality gate считает overlap как soft UX counter, не как blocker.

7. F2 `tenant config`.
   - Добавлен `tenant_config_v1` loader и текущий config для Фотона.
   - В config вынесены industry, products, CRM target/protected fields, orphan policy, privacy policy, bot boundary.
   - Gate/export summary пишут fingerprint tenant config.

8. C10 `third-party PII policy`.
   - Телефоны в AI-тексте hard-redact уже остаются включенными.
   - Для Фотона third-party names пока `warn_internal_crm`; для SaaS/bot-safe это должно стать hard-redact/block через tenant policy.

## Что v5 разрешает

- AMO real-tunnel dry-run: разрешен после прохождения v5 gate, потому что он не пишет в CRM и измеряет live match distribution.
- Live writeback: разрешается только при `Stage15 passed` + `CRM writeback quality summary passed` + `population_recall.passed_for_live=true` + per-row live guards.

## Что v5 не делает

- Не создает контакты в AMO автоматически.
- Не превращает service/existing-client звонки в non-conversation.
- Не решает entity resolution нескольких телефонов/семей.
- Не делает LLM second-pass для спорных A2/A3 случаев; это следующий слой после deterministic gate.

## Критерий следующего шага

1. Все focused tests green.
2. Пересобран post-backfill export.
3. CRM writeback quality gate `fail-live` пройден или выдал измеримый список remaining blockers.
4. Собран `audits/_inbox/amo_post_backfill_writeback_20260510_v5_product_gate` для Claude Code.
