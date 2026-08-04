> FAIL 2026-08-04 04:21 | ветка main | codex | причина: semantic/business BLOCKED: исторический платёж может стать ложным текущим подтверждением; вложенный бренд Wappi и чужой бренд за LIMIT 200 не защищены

> TAKE 2026-08-04 03:42 | ветка main | codex

Ветка: main
Зоны: src/mango_mvp/customer_timeline/bot_safe_runtime_context.py, tests/test_bot_safe_runtime_context.py, docs/worktrees_registry.md, tasks/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest --collect-only -q tests/test_bot_safe_runtime_context.py tests/test_bot_safe_direct_path_context.py
Семантический-аудит: да

# Историческая оплата в существующем семейном досье

## Цель

Выборочно поглотить бизнес-смысл ветки
`claude/timeline-final-20260803@e917db33`, не перенося её второй purchase
pipeline. Бот должен видеть только исторический признак входящего платежа точно
выбранного ребёнка. Сумма, дата, услуга, текущий доступ, возврат и бренд платежа
не выводятся.

## Инвентаризация и минимальный вариант

Механизм уже существует:
`_build_bot_safe_family_projection -> commerce.payment_history ->
_family_dossier_item -> build_bot_safe_crm_context -> direct-path prompt`.

Три варианта рассмотрены до кода:

1. прямой merge донора — отклонён: второй chunk/source/nightly path и доказанное
   смешение оплат детей;
2. фраза в `bot_safe_summary` — отклонена: дублирование и более слабая детская
   атрибуция;
3. корневой фикс существующего family dossier — выбран.

Корень: все реальные `tallanto_payment` не имеют бренда и отбрасываются до уже
существующего `_is_confirmed_payment_event`. Разрешить такой платёж только после
точной child-attribution и только если среди событий выбранного ребёнка нет
подтверждённого чужого бренда. Явно чужой бренд остаётся закрытым.

## Приёмка

1. Brandless/unknown confirmed payment выбранного ребёнка даёт
   `payment_history=fact_confirmed` и один раз доходит до финального prompt.
2. Платёж другого `customer_id` или `child_key` не доходит.
3. Чужой подтверждённый бренд ребёнка блокирует безбрендовый платёж.
4. Явно чужой бренд платежа, unmatched/low/ambiguous, superseded, non-in,
   zero/NaN/Infinity и чужой source остаются закрыты.
5. В prompt нет суммы, даты, payment/contact id и утверждения о текущем доступе.
6. Возврат старого brand-only условия красит позитивный сквозной тест; снятие
   customer/child фильтра красит изоляционный тест.
7. Никаких записей в Timeline, внешние системы и live. Новых chunk/source,
   ночных шагов, конфигов, флагов и зависимостей нет.
8. Точечные тесты, полный pytest, data/semantic/business/breaker PASS и audit
   pack. Код сам по себе не разрешает live-start.

## Данные до правки

Свежий staging измеряется только `mode=ro` + `PRAGMA query_only=1`; каждое число
фиксируется вместе с SHA и mtime. Из-за параллельной summary-only сборки финальные
числа переснять после её завершения. Предварительно top-200 ограничение теряет
только 3 из 6 100 безопасных root-контекстов, поэтому новый all-time pipeline не
строить.

## Стоп

- меняется чужой tracked-файл вне зон;
- платёж другого ребёнка или известного чужого бренда достигает prompt;
- нужен live-write, пересборка production DB или отправка клиенту;
- минимальная правка требует нового pipeline/флага без нового решения владельца.
