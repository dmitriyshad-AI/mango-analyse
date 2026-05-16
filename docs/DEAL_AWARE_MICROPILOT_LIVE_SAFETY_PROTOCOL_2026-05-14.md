# Deal-Aware Micro-Pilot Live Safety Protocol

Цель: разрешать запись deal-aware полей в AMO только после узкого, проверяемого микропилота. Этот документ не разрешает live-запись сам по себе.

## Стартовая позиция

- Stage20 preview улучшен до состояния, которое Claude счёл готовым к РОП-проверке.
- 709 строк Stage6 классифицированы: 680 dry-run, 29 blocked.
- `stage2_confidence_not_high` теперь трактуется как предупреждение старого порога, а не как hard-blocker.
- Следующий live-пилот должен быть не больше 5 сделок.

## Перед микропилотом

1. Пересобрать свежий AMO snapshot и Tallanto snapshot в день записи.
2. Взять только clean rows: одна сделка, один AMO lead, один телефон, точный Tallanto match, Stage5/Stage6 без blockers.
3. Исключить строки с оплатой/чеком, если следующий шаг может звучать как повторный дожим оплаты.
4. Исключить AMO/Tallanto mismatch, multi-phone, terminal/lost/duplicate/existing-client причины.
5. Сформировать `N <= 5` кандидатов и отдельный audit pack.
6. Получить Claude PASS/PASS_WITH_LIMITATIONS без P0/P1/P2 blockers.
7. Получить ручное подтверждение оператора на точный CSV и `expected_written=N`.

## Во время live-записи

- Писать только deal AI fields.
- Не менять статус сделки, задачи, контакты, телефоны, email, Tallanto.
- Использовать fail-fast: первая неожиданная ошибка останавливает batch.
- Перед PATCH сохранить pre-image текущих AI fields из AMO.
- В report сохранить input hash, payload hash, field ids/types, lead ids, batch id.

## После live-записи

- Сразу запустить readback gate.
- Gate должен подтвердить `evaluated=N`, `blocking_rows=0`, `failed_rows=0`, `risk_counts={}`.
- РОП вручную смотрит все `N` карточек.
- Следующая партия запрещена, пока readback и ручная проверка не зелёные.

## Rollback policy

- Rollback разрешён только отдельной командой и отдельным confirmation token.
- Rollback восстанавливает pre-image только если текущие AMO значения равны нашим expected values.
- Если менеджер уже поменял поле руками, строка уходит в manual restore, автоматический rollback запрещён.
- Нужно поддержать очистку поля, если pre-image был пустым.

## Политика AMO/Tallanto mismatch

- Для первого микропилота любой mismatch hard-block.
- В будущих партиях mismatch можно писать только как `review`, без коммерческого дожима и с явным предупреждением сверить AMO/Tallanto.
- Tallanto всегда read-only.
