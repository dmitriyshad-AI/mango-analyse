# Deal-aware Stage20 ROP iteration report — 2026-05-13

## Итог

Проведены 3 итерации из запланированных максимум 5. Остановились раньше, потому что Claude на Iter03 дал `PASS`: все 20 строк признаны `ready_for_rop`, `needs_fix_before_rop=0`, `block=0`.

Это был продуктовый ROP-precheck: проверяли не live-write, а качество текста, который увидит руководитель продаж.

## Артефакты финальной итерации

- Пакет для проверки: `audits/_inbox/deal_aware_stage20_rop_iter03_20260513/`
- Основная таблица для РОПа: `audits/_inbox/deal_aware_stage20_rop_iter03_20260513/stage20_rop_precheck_rows.csv`
- Быстрая компактная таблица: `audits/_inbox/deal_aware_stage20_rop_iter03_20260513/stage20_rop_precheck_compact.csv`
- Claude PASS: `audits/_results/2026-05-13_deal_aware_stage20_rop_iter03/CLAUDE_REAUDIT_RESULT.md`
- Row decisions: `audits/_results/2026-05-13_deal_aware_stage20_rop_iter03/row_decisions.csv`
- Findings: `audits/_results/2026-05-13_deal_aware_stage20_rop_iter03/findings.csv`

## Итерации

| Итерация | Claude verdict | Ready | Needs fix | Block | Главные проблемы | Что исправлено |
|---|---:|---:|---:|---:|---|---|
| Baseline precheck | PASS_WITH_LIMITATIONS | 13 | 7 | 0 | шаблонная сводка, generic payment next-step, payment/status conflict, raw Tallanto | сформирован план классов правок |
| Iter01 | PASS_WITH_LIMITATIONS | 14 | 6 | 0 | service-feedback routing, customer-side next-step, AMO/Tallanto mismatch | смысловая сводка, next-step из звонка, readable Tallanto, payment-paid guard |
| Iter02 | PASS_WITH_LIMITATIONS | 15 | 5 | 0 | customer-side next-step, passive wait, false service flag | manager-action rewrite, status mismatch warning, narrower service detector |
| Iter03 | PASS | 20 | 0 | 0 | только low/info замечания | summary не копирует next-step, passive/customer steps стали active manager actions, false service flag снят |

## Закрытые классы проблем

1. Шаблонная `AI-сводка по сделке` заменена на смысловую: сделка, суть, дата последнего контакта, контекст оплаты/service-feedback, количество релевантных звонков.
2. `AI-рекомендованный следующий шаг` теперь формулируется как действие менеджера, а не как обещание клиента.
3. Пассивные формулировки вроде `Ждать оплату до 15 мая` переписываются в активный контроль с fallback-действием.
4. Если есть признаки уже совершенной оплаты/чека, AI не дожимает на оплату, а просит сверить AMO/Tallanto и обновить статус.
5. Коммерческий звонок про курс/формат/стоимость больше не помечается как сервисная обратная связь только из-за слов `занятия` или `домашние задания`.
6. Service-feedback ветка теперь даёт операционное действие: передать ОС куратору/ответственному, зафиксировать ответ родителю, отдельно проверить актуальность сделки.
7. Tallanto-блок стал человекочитаемым: ученик, тип, филиал, финансы, посещения, последнее занятие, основные группы без сырого дампа и без `Текст сокращен до лимита поля`.
8. AMO/Tallanto mismatch явно подсвечивается в `AI-предупреждение по сделке`, если Tallanto показывает активного ученика при статусе сделки вроде `Перспектива`, `Ожидание оплаты`, `Заключение договора`.
9. Summary больше не копирует next-step, чтобы разные поля не заставляли РОПа читать одно и то же.
10. Нормализованы tenant-термины: `МПК/НПК/УМПК МФТИ` -> `УНПК МФТИ`, `летняя ночная школа` -> `летняя очная школа`, `унпк` -> `УНПК`.

## Проверки

- Тесты: `13 passed` по `tests/test_deal_aware_stage4_preview.py`, `tests/test_deal_aware_stage5_quality_gate.py`, `tests/test_deal_aware_stage6_writeback.py`.
- Final LOCAL_CHECK: 20 строк, 20 уникальных сделок, те же deal_id что в исходном Stage20, все `stage6_status=dry_run`.
- Bad markers final: `ellipsis=false`, `bad_mfti=false`, `night_school=false`, `old_template_summary=false`, `generic_payment_step=false`, `tallanto_truncation_phrase=false`.

## Что можно отправлять РОПу

Отправлять можно финальную таблицу Iter03, а не старую live-выгрузку:

`audits/_inbox/deal_aware_stage20_rop_iter03_20260513/stage20_rop_precheck_rows.csv`

Короткая инструкция РОПу:

> Это пилот deal-aware слоя: AI пишет сводку и следующий шаг именно по сделке, а не только по карточке контакта. Проверьте, правильно ли AI понял сделку, следующий шаг и предупреждения. Если AMO/Tallanto расходятся, AI должен явно подсветить это в предупреждении.

## Ограничение

Этот PASS не разрешает новый live-writeback. Он подтверждает качество preview для РОПа. Для записи обновлённых deal-aware полей в AMO нужен отдельный Stage6 live preflight, Claude-аудит live-пакета, операторское подтверждение и readback gate.
