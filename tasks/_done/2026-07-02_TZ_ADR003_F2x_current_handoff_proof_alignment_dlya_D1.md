Ветка: codex/adr003-semanticframe-migration
Зоны: scripts/report_adr003_frame_calibration_queue.py, tests/test_report_adr003_frame_calibration_queue.py, audits/_inbox/adr003_f2x_current_handoff_proof_alignment_*/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_report_adr003_frame_calibration_queue.py
Семантический-аудит: да

# ADR-003 F2x: current handoff + proof/source alignment report

Цель: продолжить F2b/F2w без изменения поведения бота. Отчёт должен отличать реальные текущие handoff-строки от строк, где бот уже отвечает сам, и не считать proof/text renderer готовым, если найденный факт не покрывает запрошенную ось.

Контекст: свежий регрейд 36ea110 показал, что price-route рычаг мёртв, а 14 too_cautious в основном относятся к существованию/формату курса/лагеря, не к harmless ack/status. Дополнительно аудитор нашёл измерительную дыру: один `fact_key` может существовать у двух брендов, а старый отчёт выбирал первый факт и получал ложный `wrong_brand`.

Сделано:
- `current_handoff_queue`: отдельная очередь только для строк, где текущий маршрут действительно `manager_only`/`draft_for_manager`.
- `next_autonomy_workstream`: классификация следующего безопасного шага (`fact_verification_or_retrieval_needed`, `fix_proof_axis_alignment`, `danger_adjacent_do_not_lower`, `no_current_route_leverage`, и т.д.).
- Brand-aware lookup в отчёте KB-фактов: при дубликате `fact_key` выбирается факт активного бренда, но collision сохраняется в телеметрии.
- Source-axis gate: факт про `classes` не считается покрытием для питания, медицины, охраны, живых мест, оплаты/доступа после оплаты и других чужих осей.
- Markdown-вывод с current handoff queue, source alignment и collision counters.

Что НЕ сделано:
- Не менялся runtime/provider/direct_path/profile.
- Не включалась автономность.
- Не генерировался клиентский текст.
- Не трогались live bot, Wappi, AMO/Tallanto/CRM.

Результат на сырье 36ea110:
- `too_cautious_total=14`
- `current_handoff_total=5`
- `clean_route_only_discussion=0`
- `proof_reconciliation_would_reconcile=9`
- `proof_text_shadow_renderer_candidates=0`
- `source_fact_brand_index_collisions=5`
- `source_alignment_by_status`: `aligned_covers_missing_fact_axis=2`, `blocked_source_axis_mismatch=7`
- `current_handoff_queue.by_next_autonomy_workstream`: `danger_adjacent_do_not_lower=2`, `fact_verification_or_retrieval_needed=2`, `fix_proof_axis_alignment=1`

Вердикт: report-only PASS, active autonomy NO-GO. Следующий полезный шаг - чинить proof/source alignment и доставку точного факта в тени; renderer включать рано.
