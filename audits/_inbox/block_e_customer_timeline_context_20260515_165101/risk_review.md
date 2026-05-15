# Risk Review

## Live-системы

Live write не запускался.

Нет записи в AMO, CRM, Tallanto, мессенджеры, runtime DB или `stable_runtime`.

## Основные риски

1. Неполный импорт customer timeline может дать неполную историю.
   - Снижение риска: добавлен coverage-аудит и staged promotion.

2. Timeline-контекст мог случайно попасть в AMO payload.
   - Снижение риска: контекст добавляется только в отдельные preview-поля и не входит в `DEAL_AI_FIELDS`.

3. Отсутствующая timeline DB могла ломать pipeline.
   - Снижение риска: provider возвращает fallback и warning.

## Остаточный риск

Перед включением `timeline_primary_read_enabled` нужно прогнать coverage-аудит на реальных deal-aware телефонах.
