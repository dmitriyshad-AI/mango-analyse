# CRM Text Quality Plan After Stage20 Live Writeback

Дата: 2026-05-10

## Контекст

После первого live-writeback в amoCRM был записан staged batch на 20 контактов:

- run: `stable_runtime/amocrm_runtime/contact_writebacks/20260510T141007Z/`
- результат: `20/20 written`, `0 skipped`, `0 failed`
- readback из live AMO: `20/20` карточек прочитаны обратно
- поле `Авто история общения` заполнено у всех 20
- длина поля: `731..2314` символов

Технически writeback успешен. По качеству текста выявлены UX/semantic проблемы, которые не являются P0 safety leak, но могут снижать полезность для менеджеров и масштабируемость SaaS.

## Фактические классы проблем

### Q1. Lossy ellipsis truncation

**Симптом:** внутри CRM-текста появляется `...`, например в хронологии: `Клиент...` или `не...`.

**Почему это класс, а не частный баг:** любые правила, которые обрезают содержательный текст через `...`, создают невосстановимую потерю контекста. В CRM history это нельзя считать нормальным, потому что менеджер должен видеть всю содержательную информацию или явный структурный пропуск без маскировки.

**Текущие источники в коде:**

- `scripts/build_post_backfill_amo_ready_export.py::_history_line`: обрезает `gist` до 220 символов через `...`.
- `scripts/write_amo_ready_contacts.py::_compose_last_summary`: fallback обрезает history до 252 символов через `...`.

**Целевое правило:** в полях CRM writeback не должно быть `...` как результата автоматической обрезки. Если текст слишком длинный, нужно либо:

- делать структурное сжатие без потери ключевых фактов;
- писать полную версию в textarea field;
- блокировать строку в quality gate, если поле не поддерживает длину.

### Q2. Duplicate label + count artifacts

**Симптом:** `летний лагерь | летний лагерь: 14`, `математика | математика: 3`.

**Почему это класс:** агрегатор смешивает raw-label и count-label как разные факты. На любой компании/тенанте это будет повторяться для продуктов, предметов, филиалов, источников и возражений.

**Текущий источник:**

- `scripts/build_post_backfill_amo_ready_export.py::_unique_parts` получает одновременно `Рекомендуемый продукт`, `Продукты интереса` и `chain.products_top`, где `chain.products_top` уже содержит счетчики.

**Целевое правило:** продукт/предмет должен выводиться в одном каноническом формате:

- `летний лагерь (14 касаний)` вместо `летний лагерь | летний лагерь: 14`;
- либо `Продукты: летний лагерь; Предметы: математика, физика` без счетчиков в CRM-card;
- count-сигналы можно хранить отдельно для аналитики, но не смешивать с человекочитаемой CRM history.

### Q3. Weak / stale objection labels

**Симптом:** в общие возражения попадают `время`, `доверие`, `неактуально`, `неудобно`, хотя в последних звонках клиент может быть теплым и двигаться к оплате.

**Почему это класс:** текущий слой агрегирует все исторические возражения без учета давности, силы сигнала и статуса воронки. В результате менеджер видит устаревшие/слабые негативные маркеры рядом с актуальным следующим шагом.

**Текущий источник:**

- `scripts/build_post_backfill_amo_ready_export.py::_build_contact_summary` берет `_unique_parts((row.get("Возражения") for row in contentful_desc), limit=6)` по всей цепочке.
- поле `Возражения` в AMO payload берется из всего contact row без разделения на current/historical.

**Целевое правило:** разделить:

- `Актуальные ограничения` — только последние 1-3 содержательных касания или последние N дней;
- `Исторические возражения` — отдельно, компактно, только если они не были сняты;
- `Слабые LLM-ярлыки` — `время`, `доверие`, `цена`, `неактуально`, `неудобно` не выводить сами по себе без текстового evidence.

### Q4. Next-step / priority / objection consistency

**Симптом:** карточка может иметь `warm 65%` и одновременно старое `неактуально`; или `Следующий шаг: Отменить запись` при still CRM-ready context.

**Почему это класс:** CRM-text должен быть внутренне непротиворечивым. Иначе менеджер не понимает, идти в продажу, закрывать, не беспокоить или ждать клиента.

**Целевое правило:** перед writeback запускать consistency gate:

- если next step содержит `не беспокоить`, `отменить`, `ждать обращения`, priority/probability должны быть снижены или строка должна идти в manual-review;
- если latest next step продажный (`оплата`, `ссылка`, `платежка`) — старые `неактуально` не должны быть в актуальных возражениях;
- если есть конфликт, поле должно явно писать: `Исторически было: ..., сейчас актуально: ...`.

### Q5. Manager UX length and chronology budget

**Симптом:** 6 из 20 историй длинные и включают и сводку, и хронологию. Это не ошибка, но может перегружать карточку.

**Целевое правило:** для AMO contact-card сделать два режима:

- `compact CRM card` для менеджера: 600-1200 символов, только актуальный контекст;
- `full history` для отдельного поля/файла/ссылки: полная история без потерь.

Важно: compact не должен обрезать через `...`; он должен структурно выбирать самое важное.

### Q6. Post-writeback readback gate

**Симптом:** качество стало видно только после ручного открытия карточек.

**Целевое правило:** каждый live stage должен автоматически делать readback из AMO и прогонять `crm_text_quality_gate` по фактически записанным полям. Следующий stage разрешается только если:

- `empty_auto_history=0`;
- `lossy_ellipsis_truncation=0`;
- `protected_field_hits=0`;
- `low_value_marker_in_written_history=0`;
- `duplicate_label_and_count` ниже threshold или 0;
- `weak_or_stale_objection_labels` не блокирует, но попадает в report.

## План реализации

### Phase 1. Зафиксировать class-based spec и frozen examples

1. Добавить CRM text quality section в `docs/THREAT_MODEL.md`.
2. Создать fixtures на базе stage20 readback:
   - positive examples: хорошие CRM histories;
   - blocker examples: `...`, low-value marker, пустая история;
   - warning examples: duplicate label/count, weak objections, verbose history.
3. Создать `crm_text_quality_detector.py`, независимый от builder.
4. Создать тесты для Q1-Q6.

### Phase 2. Исправить генератор текста

1. Убрать lossy `...` из CRM writeback path.
2. Переписать агрегацию products/subjects так, чтобы raw-label и count-label не дублировались.
3. Разделить `current_objections` и `historical_objections`.
4. Добавить consistency check между priority/probability/next_step/objections.
5. Ввести compact/full режимы для AMO contact card.

### Phase 3. Добавить gate перед AMO-ready и после writeback

1. `run_crm_writeback_quality_gate.py` должен блокировать Q1/P1 классы до live-writeback.
2. `write_amo_ready_contacts.py` после live stage должен уметь делать readback report.
3. Stage50/Stage86 разрешать только после green post-writeback readback gate на stage20.

### Phase 4. SaaS-readiness

1. Вынести thresholds и политики в tenant config:
   - max compact history length;
   - allow/hide chronology;
   - objection weak-label policy;
   - field capacity / textarea requirement.
2. Для новых компаний запускать detector на dry-run до первого live-writeback.
3. Не разрешать live-writeback, если target AMO fields не textarea и text может быть обрезан.

## Что делать перед stage50

До stage50 не нужно откатывать stage20: данные полезные и не содержат критичных safety-проблем. Но перед следующим stage надо:

1. реализовать Q1 минимум: запрет `...` в CRM writeback;
2. реализовать Q2 минимум: убрать `X | X: N`;
3. реализовать readback gate;
4. пересобрать stage50 и сделать dry-run;
5. отдать Claude свежий pack на audit.

## Claude Audit Refinements (2026-05-10)

Источник: `audits/_results/2026-05-10_crm_text_quality_stage20/`.

Claude verdict: `PASS_WITH_LIMITATIONS`.

### Что подтверждено

- Q1-Q6 являются правильными классами.
- Stage20 откатывать не нужно: нет P0/P1 safety leak, это manager-UX/semantic defects.
- Stage50 блокируется до исправления минимум Q1/Q2/Q4 + readback gate.

### Обязательные уточнения

1. Q1 scope должен включать `_call_gist`.
   - `_history_line` вызывает `_call_gist`, и именно оттуда появились внутренние `...` в rows 4 и 12.
   - Gate должен ловить любые `...` внутри поля, не только `value.endswith("...")`.

2. Q3 нужно разделить на два подкласса.
   - `Q3a weak filler labels`: `время`, `доверие`, `цена`, `неудобно` — warn/report, не blocker сами по себе.
   - `Q3b historical strong negative`: `неактуально`, `отказ`, `не интересно`, `не беспокоить` — нельзя смешивать с актуальными возражениями без даты/evidence; при конфликте с sales next step уводить в historical/manual-review.

3. Q4 нужно расширить до Q4b/Q4c.
   - `Q4b stale uniform recommended followup date`: нельзя всем ставить дату анализа как дату следующего контакта.
   - `Q4c vague time marker in next step`: `связаться позже`, `связаться в мае`, `вернуться при изменении решения`, `ждать обращения` не являются actionable next steps без конкретной даты/условия/ответственного.

4. Exit criterion перед stage50 должен быть жестче.
   - `crm_text_quality_detector.py` независим от builder.
   - Frozen fixtures имеют forward/random/negative-overblock layers.
   - Internal ellipsis in any target AI field = P1 fail-live.
   - Duplicate label/count = P1/P2 fail-live before stage50.
   - Closure/passive next steps force downgrade/manual-review.
   - Post-writeback readback gate green before next stage.
   - Recommended date derived from next-step semantics, not run date.
   - Tenant config holds thresholds and policies.

### Stage20 row decisions

- `allow`: 14 rows.
- `needs_review`: rows `4`, `7`, `9`, `12`, `15`, `18`, `20`.
- Не делать auto-overwrite stage20 сейчас. Эти строки можно оставить как manager-assist data, но похожие строки не должны попасть в stage50 до фиксов.

## Implementation Status 2026-05-10 19:28 MSK

Implemented as class-level controls, not literal-only fixes:

- Q1: CRM target fields now fail on internal or terminal `...` / `…`; builders compact with explicit `[сжато]` markers instead of lossy ellipsis.
- Q2: raw + counted duplicate labels are canonicalized and blocked by `crm_text_quality_detector`.
- Q3: weak filler objections are warnings; strong negative/current sales conflicts are blocking.
- Q4: passive/closure/vague next steps are routed to manual review; stale dates are not mass-promoted to the analysis date; active steps receive action-based follow-up dates.
- Q5: AMO card text is compacted; writer avoids reintroducing verbose chronology when it exceeds the UX budget.
- Q6: live-writeback requires `crm_text_quality.passed_for_live=true` and zero blocking rows in `run_crm_writeback_quality_gate.py` summary.

Latest strict artifacts:

- Export: `stable_runtime/sales_master_export_20260510_after_quality_backfill_v5_crm_text_quality_strict/`
- Gate: `stable_runtime/crm_writeback_quality_gate_20260510_v10_crm_text_quality_strict/summary.json`
- Offline preview: `stable_runtime/amocrm_runtime/contact_writebacks/20260510T162809Z/`
- Claude audit pack: `audits/_inbox/crm_text_quality_stage69_preflight_20260510_v1/`

Current measured status:

- `amo_export_ready_rows=69`
- gate `passed=true`
- `blocking_rows=0`
- `crm_text_quality.passed_for_live=true`
- offline preview `69/69`
- sanity scan: `0` ellipsis, `0` duplicate raw+count artifacts, `0` empty follow-up dates.

Claude CLI audit could not be launched from the sandbox because Claude tried to write `/Users/dmitrijfabarisov/.claude.json`, which is outside writable roots. The pack is ready for manual Claude CLI execution.

Additional hardening 2026-05-10 19:34 MSK:

- Live writeback now verifies that `run_crm_writeback_quality_gate.py` summary `input` matches the actual `--input` CSV. This prevents accidental reuse of a green summary from another staged batch.
- Regression suite after this hardening: `92 passed`.

## Claude Stage69 Audit Follow-Up 2026-05-10 20:27 MSK

Claude verdict on `audits/_inbox/crm_text_quality_stage69_preflight_20260510_v1`: `PASS_WITH_LIMITATIONS`.

Key actionable limitation: rows 16, 23, 67 in the 69-row strict export are review-precision marker cases (`rv_learning_not_discussed`). Claude manually judged them likely false-positive ed-tech narratives, but recommended manager confirmation before counting them in live writeback.

Implemented safer next input without changing the canonical 69-row export:

- Stage66 live candidate CSV: `stable_runtime/sales_master_export_20260510_after_quality_backfill_v5_crm_text_quality_strict/amo_export_ready_stage66_without_review_marker_ru.csv`
- Stage66 XLSX: `stable_runtime/sales_master_export_20260510_after_quality_backfill_v5_crm_text_quality_strict/amo_export_ready_stage66_without_review_marker_ru.xlsx`
- Review-marker rows CSV: `stable_runtime/sales_master_export_20260510_after_quality_backfill_v5_crm_text_quality_strict/amo_export_ready_stage69_review_marker_rows_ru.csv`
- Review-marker rows XLSX: `stable_runtime/sales_master_export_20260510_after_quality_backfill_v5_crm_text_quality_strict/amo_export_ready_stage69_review_marker_rows_ru.xlsx`
- Stage66 gate: `stable_runtime/crm_writeback_quality_gate_20260510_v11_stage66_no_review_markers/summary.json`
- Stage66 offline preview: `stable_runtime/amocrm_runtime/contact_writebacks/20260510T172714/`

Measured Stage66 status:

- rows: `66`
- gate `passed=true`
- `blocking_rows=0`
- `crm_text_quality.passed_for_live=true`
- `population_recall.passed_for_live=true`
- population marker hits: `0`
- offline preview: `66/66`, `0` skipped, `0` failed

Next operational sequence:

1. Raise AMO runtime DB tunnel.
2. Run real-tunnel dry-run on the Stage66 CSV with Stage66 quality summary.
3. If dry-run is green, write first staged live subset (20 rows), then run post-writeback readback gate.
4. Only after readback is green, continue the next staged live subset.
5. Keep rows 16/23/67 in manager-confirmation queue; add them later only after confirmation or refined detector policy.
