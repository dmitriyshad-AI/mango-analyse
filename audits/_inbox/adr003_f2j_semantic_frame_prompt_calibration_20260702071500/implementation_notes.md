# ADR-003 F2j Implementation Notes

## Что изменено

- В inline `SemanticFrame SHADOW` prompt добавлена явная граница:
  - “есть ли курс/лагерь/формат/подходит ли класс” = `answer_question`;
  - “есть места/можно попасть/бронь/лист ожидания” = `check_availability`.
- В posthoc SemanticFrame prompt добавлена та же граница.
- Добавлены regression tests на оба prompt-а.

## Что не изменено

- Поведение бота.
- Route/text.
- Profile/default flags.
- P0-floor/preblock.
- Direct path gates.
- Manager-only policy.

## Локальный measurement

Прогон: `semantic-frame-enrich-from` на 7 диалогах из F2i workstream `semanticframe_existence_vs_availability`, без Telegram/AMO/CRM/Tallanto.

Результат:

- `requested_action_wrong`: 6 -> 1;
- `check_availability`: 6 -> 0;
- `must_handoff_wrong`: 8 -> 8.

Интерпретация: prompt-калибровка исправляет часть `requested_action`, но не решает `risk_class/answerability/must_handoff`.

## Вердикт

F2j можно считать частичным progress, но активное понижение маршрута по-прежнему запрещено.
