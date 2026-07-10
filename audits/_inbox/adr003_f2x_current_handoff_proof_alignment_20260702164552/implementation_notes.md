# ADR-003 F2x Current Handoff + Proof Alignment

Изменён только отчётный скорер `scripts/report_adr003_frame_calibration_queue.py` и его тесты.

Добавлено:
- очередь `current_handoff_queue`, чтобы не считать уже самостоятельные ответы рычагом автономности;
- поле `next_autonomy_workstream` для каждого too-cautious примера;
- brand-aware lookup по KB при дубликатах `fact_key`;
- source-axis проверка, которая блокирует renderer, если найденный факт не покрывает запрошенные missing-facts.

Поведение бота, маршруты, текст, флаги и профиль не менялись.
