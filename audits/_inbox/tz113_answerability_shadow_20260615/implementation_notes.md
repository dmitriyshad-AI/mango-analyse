# TZ-113 answerability shadow

Дата: 2026-06-15
Ветка: codex/tz113-answerability-shadow

## Что реализовано

- Добавлен флаг `TELEGRAM_ANSWERABILITY_SHADOW`, по умолчанию выключен.
- Флаг не входит в `pilot_gold_v1` и включается только явно через окружение или контекст.
- При включенном флаге прямой путь просит модель дополнительно вернуть наблюдательные поля:
  - `can_answer_self`
  - `self_missing_facts`
  - `supporting_facts`
  - `why_manager`
- Эти поля сохраняются в `metadata["answerability_self"]` и не влияют на `route` или `draft_text`.
- Добавлен `metadata["answerability_trace"]`: свод причин из уже существующих слоев direct-path, смысловой проверки и финального защитного гейта.
- В `dynamic_summary.json` добавляется агрегат `answerability_trace`, но только если в ходе реально был такой след.

## Инварианты

- При выключенном флаге нет блока в промпте, нет `answerability_self`, нет `answerability_trace`.
- Маршрут считается только из `payload["route"]`.
- P0, брендовые проверки, финальный гейт и смысловой проверяющий слой не менялись.
