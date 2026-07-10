# Обратная совместимость

- Форматы: существующий runner и schema JSONL не менялись. Новый фокусный набор использует существующий формат `simulator_spec` + `judge_spec` + `persona`.
- Потребители: `roles_read/refund_tax` остаётся explicit-env only; профиль `pilot_gold_v1` не получил новых классов. B-нога M1 остаётся profile-as-is.
- Совместимость поведения: позитивный tax false-positive по `theme:009_refund` сохраняется, real refund/payment dispute остаётся менеджерским.
