> DONE 2026-08-06 04:43 | ветка codex/timeline-tallanto-money-owner-20260806 | codex

> TAKE 2026-08-06 03:22 | ветка codex/timeline-tallanto-money-owner-20260806 | codex

Ветка: codex/timeline-tallanto-money-owner-20260806
Зоны: scripts/import_tallanto_payments_to_timeline.py, tests/test_import_tallanto_payments_to_timeline.py, docs/worktrees_registry.md, docs/MANGO_CURRENT_STATE_V7.md, docs/DECISIONS_LOG.md, tasks/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_import_tallanto_payments_to_timeline.py tests/test_customer_timeline_store.py tests/test_customer_timeline_tallanto_attendance_import.py tests/test_customer_timeline_family_graph.py tests/test_customer_timeline_stage5_money_ingest.py
Семантический-аудит: да

# ТЗ: деньги Tallanto используют единственного канонического владельца student ID

## Проблема

В свежей staging 2 177 платежей и 945 абонементов имеют пустой `customer_id`,
хотя 3 108 из 3 122 событий уже содержат точный `record.contact_id`. Импортёр
денег повторно вычисляет владельца через `authoritative_exact_identity_rows()` и
блокирует точный student ID при любом семейном/контактном конфликте клиента.
Канонический `authoritative_tallanto_student_owners()` уже реализует более точное
правило и используется attendance/family graph, но money importer его не зовёт.

## Образ результата и бизнес-польза

Историческая оплата и абонемент принадлежат тому же ученику/семье, которому
Tallanto выдал точный student ID. Общий семейный телефон, старая контактная
неоднозначность или конфликт родственника не скрывают подтверждённые деньги.
Только два точных владельца одного student ID или прямой конфликт этого student
ID оставляют событие в карантине. Менеджер видит реальную историю оплат семьи,
а не пустую память.

## Минимальное решение

1. В `load_tallanto_customer_lookup()` использовать существующий
   `authoritative_tallanto_student_owners()` как единственного решателя.
2. Нижние exact rows читать только для match-class и списка кандидатов в отчёте;
   не принимать по ним второе решение о владельце.
3. Переиспользовать существующий retry локальных unowned events; Tallanto API и
   новый загрузчик не нужны.
4. Не закрывать и не удалять семейные/identity-конфликты массово. Существующий
   resolver может закрыть только прежний money-conflict того же `source_ref`
   после успешной точной перепривязки. Не угадывать события без опоры.

## Приёмка

1. Два авторитетных владельца одного student ID и прямой
   `tallanto_identity_conflict:tallanto_student_id:<id>` остаются без владельца.
2. Один точный владелец при широком family/contact conflict привязывается.
3. Manual и strong_unique сохраняют свой match-class; события без contact ID не
   угадываются.
4. На APFS-клоне лестница `1 -> 10 -> все`: заранее записано ожидание; баланс
   `получено = связано + прямой конфликт + противоречие direct/abonement owner +
   нет опоры`; повтор не создаёт дублей.
5. После полного локального retry измерены платежи/абонементы без владельца,
   customer_purchases, prompt money context и COUNT первичных событий.
6. Никаких внешних запросов/записей; `product_data` не меняется.

## СТОП

- Первый из нескольких точных владельцев выбирается автоматически.
- Общий phone/email используется вместо точного Tallanto student ID.
- Входящие и исходящие ноги денег складываются в одну бизнес-сумму.
- Нужна запись в Tallanto/AMO/Wappi или рабочую Timeline.

## Бритва

Один production-файл, без новых модулей, таблиц, флагов и зависимостей. Цель —
удалить дублированное решение, а не добавить ещё один фильтр. Бюджет: net deletion
или максимум 20 новых нетестовых строк при доказанной необходимости.

## Фактический результат на фиксированном APFS-клоне

- лестница `1 -> 10 -> все`: `1/1`, `10/10`, затем 3 110 точных связей;
- 16 старых событий с теперь конфликтным exact ID возвращены в карантин;
- итог: 113 200 owned/canonical match, 0 mismatch, 45 объяснённых unowned;
- второй проход: `45 -> 45`, только duplicate writes, первичных событий 486 114;
- `customer_purchases_v1` пересчитана, две устаревшие fact-строки удалены;
- `quick_check=ok`, `foreign_key_check=0`;
- смена единственного exact-владельца `A -> B` перепривязывает локальное событие;
- 229 целевых тестов зелёные; полный набор: 4 958 passed и 10 известных
  несмежных падений ADR-003/KB;
- prompt money для восстановленных клиентов остаётся 0 из-за независимых
  открытых `ambiguous_identity`; гейт не ослаблялся и конфликты не закрывались
  по факту оплаты.
