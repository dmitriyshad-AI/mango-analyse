# ADR-003 F2w: shadow text renderer readiness

Дата: 2026-07-02

## Задача

После F2v стало ясно: proof/reconciliation может найти точный факт, но route-only не готов, потому что текущий текст часто остаётся manager/deferral, а сырой `client_safe_text` нельзя дословно отправлять клиенту.

Нужно было проверить следующий слой строго в тени: можно ли по доказанному факту собрать безопасный текстовый кандидат без runtime/provider/direct_path изменений.

## Сделано

- В `scripts/report_adr003_frame_calibration_queue.py` добавлена report-only сводка `shadow_text_renderer_*`.
- Renderer не экспортирует полный кандидат в JSON/Markdown: только статус, blockers, source, length и SHA-256.
- Runtime, профиль, live, Telegram, AMO/Tallanto/CRM не трогались.
- Разрешён только очень узкий класс shadow-кандидата: атомарный `structured_value.classes_raw/classes` для `course_parameter`.
- Запрещено использовать как клиентский текст:
  - `client_safe_text` дословно;
  - `template_text`;
  - длинный `raw_value`;
  - wrong-brand, stale, PII, internal/forbidden facts;
  - `bot_template_required` без отдельного renderer.

## Сырой результат на 36ea110

Вход:

`audits/_inbox/adr003_f2v_existing_frame_proof_replay_20260702155709/reports_36ea110/enriched/dynamic_dialog_transcripts.jsonl`

Отчёт:

`audits/_inbox/adr003_f2w_shadow_text_renderer_20260702162937/reports_36ea110/adr003_frame_calibration_queue_report.json`

Итоги:

- `proof_reconciliation_would_reconcile=9`
- `proof_text_shadow_renderer_candidates=0`
- `proof_reconciliation_send_as_is_review_candidates=0`
- `fact_gated_strict_f3_draft_candidates=0`
- `shadow_text_renderer_by_status`:
  - `blocked_wrong_brand=5`
  - `blocked_unsupported_structured_value=3`
  - `blocked_template_renderer_not_implemented=1`

## Вывод

F2w подтверждает NO-GO для активного включения. Даже после proof-replay на свежем M1 нет безопасного текстового кандидата для самостоятельного ответа.

Следующий реальный рычаг:

1. исправлять retrieval/brand alignment для wrong-brand exact proof;
2. проектировать отдельный template renderer для `bot_template_required`;
3. расширять атомарные structured-value renderers только по одному типу факта с semantic review.

Route-only и текущий текстовый слой включать нельзя.
