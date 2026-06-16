# TZ-113 review scope

Проверь реализацию теневой ответуемости.

## Главное

1. `TELEGRAM_ANSWERABILITY_SHADOW` default OFF.
2. Флаг не включается через `pilot_gold_v1`.
3. При OFF нет новых prompt-полей, `answerability_self`, `answerability_trace`.
4. При ON модель возвращает самооценку в том же direct-path вызове, без третьего LLM-вызова.
5. `answerability_self` не влияет на `route` и `draft_text`.
6. `answerability_trace` собирает причины из существующих metadata-слоев, не пробрасывает причины между слоями и не меняет гейты.
7. `dynamic_summary.json` получает агрегат только при наличии следа.

## Проверки

См. `test_output.txt`.
