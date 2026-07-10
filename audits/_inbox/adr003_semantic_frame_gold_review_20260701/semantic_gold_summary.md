# ADR-003 SemanticFrame Gold Review Summary

Дата: 2026-07-01.

## Scope

Ручной смысловой разбор очереди `audits/_inbox/adr003_semantic_frame_gold_queue_full131_20260701/`.

Очередь построена из full131 paired-enriched transcript после фикса измерителя:

- `manager_approval_required` больше не считается route-handoff;
- route-handoff = `bot_route in {manager_only, draft_for_manager}`;
- `frame.must_handoff` требует строгий bool;
- `input_status=all_framed`;
- очередь не коммитится целиком, потому что содержит тексты диалогов.

## Aggregate Manual Read

Из 75 строк очереди:

- `frame_correct`: 29
- `frame_too_cautious`: 43
- `unclear`: 3

Разбивка по типам:

| Type | Rows | Frame correct | Too cautious | Unclear |
|---|---:|---:|---:|---:|
| `frame_self_current_handoff` | 3 | 3 | 0 | 0 |
| `frame_handoff_no_p0_signal` | 37 | 17 | 17 | 3 |
| `frame_handoff_current_self` + combined | 35 | 9 | 26 | 0 |

## What Passed Semantically

SemanticFrame полезен как смысловой сигнал для `manager_action`, особенно там, где regex/P0 не должен быть единственным критерием:

- клиент просит администратора связаться;
- запись, лист ожидания, бронь, закрепление места;
- чек/оплата/письмо требуют сверки;
- конкретное расписание завтра отсутствует или не видно;
- персональный подбор преподавателя/группы;
- наличие мест и живой статус группы.

Самые опасные найденные классы, которые текущий self-route может пропустить:

- `wappi_pair_missing_72h_005` — клиент дал контакт и просит администратора по записи/оформлению/оплате;
- `wappi_pair_missing_72h_022` — клиент прислал чек, нужна сверка оплаты;
- `tz147_ft_benign_schedule_missing_01` — занятие завтра, расписания нет;
- `ra1_foton_dates_and_booking` — бронь места;
- `wappi_pair_missing_72h_012` — оплата и письмо требуют проверки.

## What Failed Semantically

SemanticFrame пока слишком широко поднимает `must_handoff` на безопасных справочных или next-step вопросах:

- обычные цены, адрес, платформа, формат;
- порядок записи без фактической записи;
- справка по рассрочке или оплате за год;
- pre-sale refund policy без реальной претензии;
- пауза клиента: "подумаем", "оплачу позже";
- safe deferral по неподтверждённой детали лагеря;
- существование группы по расписанию, если не спрашивают свободное место.

Главная ошибка frame: смешивает `missing_facts`, `answer_question`, `check_availability` и настоящий `manager_action`.

## Decision

`formal_pass`: да, для технического paired shadow/no-op.

`semantic_pass`: нет, для active decision policy.

`active_behavior_allowed`: false.

Причина: frame полезен, но не калиброван. Если сейчас дать ему рулить маршрутом, он исправит часть опасных self-answer, но одновременно вернёт много лишних handoff на справочных вопросах.

## Next Design Constraint

Следующий active-этап должен быть узким:

1. Не "frame рулит всем".
2. Сначала только `manager_action_gate` для операций, где нельзя отвечать без реальной проверки:
   - booking/place reservation;
   - payment/check/receipt confirmation;
   - explicit administrator callback;
   - missing near-term schedule for enrolled/paid customer;
   - personalized teacher/group assignment;
   - live availability/free seats.
3. Отдельно сохранить self-answer для verified factual questions:
   - price/address/platform/format;
   - general enrollment steps;
   - installment explanation;
   - pre-sale refund policy;
   - warm acknowledgement/closing.
4. `missing_facts` не равно `must_handoff`: это должно идти в fact-gate/safe deferral, если вопрос не требует действия менеджера.
5. Любой active-этап должен идти за новым флагом default OFF и A/B against this queue.

## Required Regression Gates

- Добавить gold labels из этой очереди в версионированный eval перед включением поведения.
- Метрика acceptance для active-этапа: исправляет manager-action пропуски без роста handoff на safe factual/next-step контролях.
- P0-пол отдельно: payment/refund dispute остаётся manager-only, но benign presale/payment explanation не должен становиться P0.
