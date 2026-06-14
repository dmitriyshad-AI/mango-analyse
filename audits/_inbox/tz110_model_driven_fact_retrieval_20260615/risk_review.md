# Risk review

## Что не изменилось

- При обоих новых флагах OFF выбор фактов остается прежним.
- `TELEGRAM_LLM_RETRIEVE=0` по-прежнему не вызывает ретривер.
- `pilot_gold_v1` не включает новые флаги.
- Сквозной `required_fact_keys` для защитных слоев не менялся.
- Output gate, P0 floor, brand guard и number guard не ослаблялись.

## Основной остаточный риск

`TELEGRAM_RETRIEVER_NEED_SHADOW=1` меняет prompt ретривера, потому что модель должна вернуть дополнительную декларацию `needed_facts`. Без второго LLM-вызова нельзя формально гарантировать, что модель выберет точно те же `exact_ids/adjacent_ids`, что и в id-only prompt.

Снижение риска:

- флаг default OFF;
- prompt явно запрещает менять `exact_ids/adjacent_ids` из-за `needed_facts`;
- выбор и декларация логируются в `dynamic_summary.json`;
- ночной замер должен сравнить OFF vs A-only vs B на реальных сценариях.

## Что проверено тестами

- новые флаги default OFF даже при `pilot_gold_v1`;
- OFF сохраняет id-only contract;
- A-only логирует декларацию без изменения selection на стабильном payload;
- B не отправляет `required_fact_keys` в prompt ретривера;
- B сохраняет keyword keys в metadata для гейтов;
- B требует `needed_facts`, иначе откатывается в keyword fallback;
- scope conflict exact -> adjacent логируется в `scope_demoted_ids`;
- `dynamic_summary.json` содержит машинный `fact_retrieval_trace`.
