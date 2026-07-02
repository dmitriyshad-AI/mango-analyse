# Risk Review

## Основные риски

1. **Ложный вывод "можно включать route-only"**
   - Снижен: отчёт явно показывает `harmless_context_ack_status_candidates=0` и `draft_candidates_for_future_active=0`.
   - Остаточный риск: человек может смотреть только на `frame_too_cautious=14` и пропустить, что 8 строк требуют fact-verification.

2. **Выдумка существования курса/формата**
   - Снижен: `existence_format_needs_fact_verification_blocked` и `frame_too_cautious.existence_format_count` выделены отдельно.
   - Остаточный риск: следующий исполнитель может попытаться решить это маршрутом, а не проверкой фактов.

3. **Понижение `manager_only`**
   - Снижен: в ТЗ и отчёте `manager_only` только диагностируется.
   - Остаточный риск: любое будущее изменение этого инварианта требует отдельного ADR/да Дмитрия.

4. **PII в audit pack**
   - Снижен: отчёт пишет redacted/truncated excerpts; тест проверяет phone/email/id.
   - Остаточный риск: исходные M1-транскрипты остаются локальным сырьём и не копируются этим блоком.

## Что не трогалось

- Live Telegram bot.
- Wappi observe.
- AMO/CRM/Tallanto.
- Профильные флаги.
- P0 floor/preblock.
- Runtime direct-path поведение.
