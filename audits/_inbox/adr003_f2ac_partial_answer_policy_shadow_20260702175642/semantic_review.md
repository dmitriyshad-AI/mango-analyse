# Semantic review

Verdict: `PASS_WITH_NOTES`

## Что прошло

- Отчет не генерирует и не экспортирует клиентский текст.
- Оба F2z partial-кандидата проверены не изолированно, а через calibration queue.
- `p0_model_led_pos_how_next#1` заблокирован как `danger_adjacent_do_not_lower`.
- `wappi_pair_missing_72h_002#1` заблокирован как `source_axis_mismatch`.
- Active readiness остается `no_go`.

## Риски

- Это смысловой стопор, а не улучшение автономности.
- Нельзя использовать `draft_partial_shadow_candidate` из F2z без этого join:
  иначе возникнет ложное ощущение готовности к частичным ответам.

## Требуемое правило дальше

Любой будущий partial-answer/text слой должен сначала проходить этот тип
склейки с blockers и отдельный semantic review. Нельзя включать текст только по
наличию partial support.
