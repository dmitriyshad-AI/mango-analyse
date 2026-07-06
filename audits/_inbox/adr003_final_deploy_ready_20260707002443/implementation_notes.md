# Что сделано

- Доведен PaymentFix-хвост #16: оплачено/чек/доступ/занятие без явного запроса возврата теперь получает `payment_dispute`-текст, а не возвратный шаблон.
- Закреплен инвариант `correct_route_wrong_p0_text`: при валидном SemanticFrame legacy/P0/text-hygiene не выбирает смысловой клиентский шаблон `refund/tax/payment_dispute`; он может только ужесточить маршрут, а конфликт уходит в trace.
- Добавлен нейтральный P0-text guard: если frame уверенно говорит `risk_class=p0`, `must_handoff=true`, `answerability=manager_only`, текст заменяется короткой передачей менеджеру без консультации и сбора деталей.
- Закрыт edge-case аудита: тот же валидный P0-frame принудительно ужесточает route до `manager_only`, даже если legacy-P0 metadata пустая.
- Настоящие refund/dispute оставлены manager-only и не превращаются в оплатный/налоговый текст.
- Добавлены регрессии на paid-no-access, receipt-not-credited, paid-lesson-missing, short follow-up через P0 latch, настоящий refund, tax-vs-refund и P0 child-safety текст.
- Синхронизирован frozen moratorium snapshot/budget с текущей веткой; новый фикс не добавляет regex-понимание, но snapshot теперь отражает уже существующие изменения ветки.

# Как проверялось

- `152 passed, 622 deselected` по P0/payment/text-hygiene/provider/moratorium срезу.
- Локальный deploy smoke, профиль `pilot_gold_v1`, `PYTHONPATH=src`, без live-write:
  - `paymentfix_after_p0_text_fix`: 20 dialogs / 40 turns, PASS=9, PASS_WITH_NOTES=11, FAIL=0, hard_gate_failures=0.
  - `p0_micro_final`: 11 dialogs / 12 turns, PASS=8, PASS_WITH_NOTES=3, FAIL=0, hard_gate_failures=0.
- Смысловая выборка по сырью:
  - `payfix_neg_foton_paid_no_access_01`: нет слова "возврат", маршрут `manager_only`, текст про сверку оплаты.
  - `payfix_neg_foton_receipt_not_credited_01`: нет обещания зачесть/открыть курс/вернуть деньги.
  - `payfix_neg_unpk_refund_after_bad_access_01`: настоящий refund остается manager-only refund.
  - `p0_model_led_neg_child_left_alone`: P0 распознан, нет консультации и сбора деталей.

# Что осталось

- Live-свап не выполнялся. Следующий шаг: регрейд audit pack/сырья и отдельное человеческое решение на swap.
- Replay full exam на M1 не запускался в этом блоке.
