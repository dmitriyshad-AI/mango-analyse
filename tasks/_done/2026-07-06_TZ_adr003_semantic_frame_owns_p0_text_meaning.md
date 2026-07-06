> DONE 2026-07-06 14:16 | ветка codex/adr003-semanticframe-migration | codex

> TAKE 2026-07-06 13:54 | ветка codex/adr003-semanticframe-migration | codex

Ветка: codex/adr003-semanticframe-migration
Зоны: src/mango_mvp/channels/, tests/, docs/, tasks/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_direct_p0_text_hygiene.py tests/test_subscription_llm_draft_provider.py
Семантический-аудит: да

# TZ: ADR003 SemanticFrame owns P0/text meaning

Контекст: решение владельца и D1/Fable 2026-07-06. Цель не в том, чтобы точечно починить НДФЛ или paid-no-access, а закрыть архитектурный остаток regex-лазаньи: при валидном inline SemanticFrame старые legacy/P0/text-hygiene слои не должны выбирать клиентский смысловой шаблон.

## Цель

Если есть валидный inline `SemanticFrame` с высокой уверенностью, то legacy/P0/text-hygiene слой:

- не имеет права выбирать клиентский смысловой шаблон `refund` / `tax` / `payment_dispute` / соседние денежно-возвратные шаблоны;
- может только усилить маршрут до менеджера;
- при конфликте frame vs deterministic floor сохраняет безопасный маршрут, но текст становится нейтральным менеджерским, без ложного refund/tax/payment смысла;
- пишет конфликт в trace/metadata;
- покрывается регрессией `correct_route_wrong_p0_text`.

## Обязательные кейсы

- `Возврат НДФЛ оформляете?` не должен получать возвратный шаблон "приняли обращение по возврату".
- `Оплатили курс, ссылка не пришла / расписание не появилось / занятие не назначили` не должен получать возвратный шаблон.
- Реальные refund/dispute остаются manager-only и не маскируются как налоговый/нейтральный безопасный self-answer.
- При низкой уверенности или мусорном frame старый fail-closed маршрут сохраняется.

## Запреты

- Не менять live runtime, AMO, Tallanto, CRM, Wappi.
- Не отправлять сообщения клиентам.
- Не удалять safety floors.
- Не добавлять новые понимающие regex как основной источник смысла. Детерминированный слой может только страховать маршрут/нейтрализовать текст.

## Проверки

- Точечные unit-регрессии для `correct_route_wrong_p0_text`.
- Existing direct P0/text hygiene tests remain green.
- Semantic review note: объяснить, почему клиентский текст стал безопаснее и где остался риск.
