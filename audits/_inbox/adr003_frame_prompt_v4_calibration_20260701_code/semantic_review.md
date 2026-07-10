# Semantic Review

Статус: `semantic_pass_for_phase1_shadow_calibration`, не `production_ready`.

Ф1-калибровка улучшила качество SemanticFrame на gold-75:

- initial: `must_handoff_accuracy=0.6027`, `too_cautious=29`, `too_confident=0`, `answerability=1/75`.
- v4: `must_handoff_accuracy=0.9315`, `too_cautious=5`, `too_confident=0`, `answerability=68/75`.

Техническая нейтральность paired enrichment:

- 131 диалог / 241 ход.
- `route_text_diff_count=0`.
- ON calls: `bot_semantic_frame_shadow=241`, все остальные bot/client/judge/memory calls = 0.
- Required frame fields complete: 241/241.

Смысловой вывод:

- Frame стал пригоднее как источник shadow-аналитики для следующей Ф2.
- Активное понижение route ещё не разрешено: осталось 5 too-cautious строк, `requested_action=62/75`, нужен регрейд Claude #1 и class-specific Ф2 shadow.
- Деньги/оплата/чек/рассрочка/списания/бронь/живые места удержаны как manager-only; `too_confident=0` на gold-75.

