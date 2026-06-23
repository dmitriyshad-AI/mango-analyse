# Risk Review

- Live/CRM/AMO/Tallanto/stable_runtime не трогались.
- Код бота не менялся; флаг `TELEGRAM_FACT_VENUE_SCOPE` не добавлялся в профиль и не включался.
- Главный риск следующей фазы: перепрунинг `any` или generic offline-фактов. В текущей фазе такие факты оставлены `any`.
- Регекс по `client_safe_text` не использовался для вывода venue.
