# Semantic review

## Verdict

PASS_WITH_NOTES.

Формально код и тесты проходят. Смысловой статус: таблица годится как manager-review preview для сверки архитектором/Дмитрием, но не как разрешение на запись в AMO.

## What passed

- CRM-текст не выдаёт неопределённые связи за точные: ambiguous/shared/family identity переводится в blocker и `Готово=нет`.
- Сделочные поля не строятся как готовые к записи при неоднозначной привязке.
- Вероятность продажи не выводится из догадок: берётся только из старого analyze fallback.
- Возражения не исчезают при сжатии: используется явный маркер `[сжато]`.
- Суммы/платежные факты остаются manager-only; bot-safe поля для CRM-проекции не заявлены.
- Бренд помечается информативно и не используется как стоп-гейт менеджерской карточки.
- Workbook явно показывает, что пойдёт в AMO, что уже есть в AMO, готовность и blockers.

## Blocking issues

- Для реальной записи все 200 строк заблокированы: AMO contact/lead ids в `read_api` маскированы или отсутствуют.
- Пока нет немаскированного approved join-key, P9 должен оставаться fail-closed. Это ожидаемый стоп-гейт, не дефект xlsx.

## Non-blocking risks

- Тексты summary/next_step зависят от качества старого analyze и customer_timeline summaries; это preview для менеджера, не клиентский ответ.
- `what_already_in_amo` ограничен тем, что `read_api` отдаёт в профиле; полноценный before-write snapshot нижнего helper будет нужен в Этапе 2.
- В workbook нет live readback из AMO, потому что live/writeback запрещены в текущем блоке.

## Missing checks

- Архитектор должен регрейдить на сырье, особенно строки с masked AMO ids, ambiguous identity и open conflicts.
- Не проверялась реальная live-запись в AMO/Tallanto: она запрещена условиями задачи.

## Required regression tests or gates

- `tests/test_crm_card_aggregator.py`:
  - пустой timeline -> fallback из analyze, не молчит;
  - повторная генерация идемпотентна по snapshot;
  - ambiguous identity -> `Готово=нет` и blocker;
  - длинное возражение -> `[сжато]` и лимит.
- `tests/test_amo_writeback_guards.py`:
  - aggregator OFF -> старое поведение;
  - contact card ON -> AI-поля добавляются только за флагом;
  - `crm_card_not_ready` блокирует live row;
  - P8 allowlist запрещает телефон/ФИО/email/ручную историю.
- `tests/test_deal_aware_structured_objections.py`:
  - explicit compaction marker появляется только за флагом.

## Recommended next action

Отдать xlsx и audit pack на регрейд Claude. До решения по немаскированным AMO join-key не переходить к live writeback.
