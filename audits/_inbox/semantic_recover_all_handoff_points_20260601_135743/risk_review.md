# Risk review

## Основные риски

1. Over-autonomy: бот мог начать отвечать соседним фактом вместо честного handoff.
   - Митигировано точным scope-check, блоком P0/refund/complaint/payment-dispute/legal и NEG-тестами.

2. Brand leakage: retrieved fact может содержать чужой бренд.
   - Митигировано существующим hard-check; добавлен тест с текстом УНПК при active_brand=Фотон.

3. Payment substitution: вопрос про прямой перевод мог получить Т-Банк как соседнюю справку.
   - Митигировано `_secondary_fact_text`: платежная вторичная справка теперь только при совпадении payment-target anchors.

4. Semantic unavailable path: recover при недоступном critic может быть опасен.
   - Митигировано cite-only candidate + `_hard_check` + запрет новых конкретных якорей без доступной семантики.

## Непокрытый остаток

- Динамический эффект на долю over-handoff и возможные редкие wrong-scope случаи не измерялся в этом коммите. Нужен следующий прогон на утвержденном наборе и независимый регрейд.
