# Обратная Совместимость

- Форматы: структура `SubscriptionDraftResult` не менялась.
- Флаги: новые флаги не добавлены; логика работает внутри существующего `TELEGRAM_TEXT_HYGIENE_PAYMENT_FIX` / профиля `pilot_gold_v1`.
- Маршруты: `manager_only` не понижается. При конфликте текст меняется на безопасный, маршрут остается ручным или ужесточается.
- Legacy/floor: P0, money, legal, brand floors не удалялись.
- Тестовые наборы: добавлен guard на существующий `adr003_acceptance_paymentfix_20260704.jsonl`; сами runtime-сценарии не изменялись.
