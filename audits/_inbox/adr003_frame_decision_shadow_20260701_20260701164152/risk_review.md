# Риски

- Клиентский риск: низкий для этого шага. Новый слой default-OFF; при ON пишет только metadata и не меняет клиентский текст/маршрут.
- P0-риск: P0-preblock, `codes_from_text`, `classify_answer_safety`, model-P0 route, P0 text hygiene и authoritative gate не менялись.
- Данные/записи: нет внешних записей, AMO/Tallanto/CRM не трогались, live bot не трогался.
- Откат: удалить вызов `apply_semantic_frame_decision_shadow`, helper/flag/export и summary-блок симулятора; при OFF поведение и так no-op.
- Остаточный риск: alignment не равен бизнес-истине, он показывает расхождение frame с текущими детекторами. Для решения о включении нужны M1/Claude semantic regread и gold-разметка.
