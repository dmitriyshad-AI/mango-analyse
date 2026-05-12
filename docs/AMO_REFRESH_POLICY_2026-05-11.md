# AMO refresh policy for already-written contacts

Дата: 2026-05-11
Scope: 40 already-written refresh candidates из `amo_waiting_autonomous_work_20260511_v1`.

## Почему refresh нужен

20+20 контактов уже были записаны раньше. После улучшения CRM text quality и post-backfill слоя их можно обновить, но только если новое значение реально лучше и безопаснее. Это не новый импорт и не массовая перезапись.

## Политика refresh

Refresh разрешается только когда выполнены все условия:

1. Строка уже была записана в AMO ранее и есть writeback report.
2. Есть успешный readback текущего значения из AMO.
3. Новый payload отличается от текущего значения по разрешенным полям.
4. CRM quality gate passed на точном refresh CSV.
5. Real-tunnel dry-run прошел без failed/protected-field/write-target нарушений.
6. Нет duplicate/multi-contact/mismatch статуса.
7. Есть explicit approval на конкретный refresh batch.
8. После live refresh выполнен readback gate.

## Разрешенные поля

Только текущие AI-поля manager-assist режима:

- `Статус матчинга`
- `AI-приоритет`
- `AI-рекомендованный следующий шаг`
- `Последняя AI-сводка`
- `Авто история общения`

Запрещено менять `Id Tallanto`, `Филиал Tallanto` и любые не-AI бизнес-поля.

## Текущие числа

- Refresh candidates: 40.
- Missing readback rows: 15.
- Contact-id mismatch rows: 1, остаются blocked.
- Recommended first live refresh stage: canary 5-10 rows after successful dry-run and audit.

## Что делать после сотрудников

Когда сотрудники сообщат, что дубли объединены:

1. Запустить after-staff recheck.
2. Пересобрать queue.
3. Снова отделить new candidates от refresh candidates.
4. Не смешивать post-merge new writes и already-written refresh в одном live stage.
