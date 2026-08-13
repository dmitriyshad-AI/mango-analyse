# Архитектура Mango

## Живые пути

```text
Wappi
  -> src/mango_mvp/integrations/draft_loop.py
  -> SubscriptionLlmDraftProvider.build_draft()
  -> SubscriptionLlmDraftProvider._build_direct_path_draft()
  -> subscription_llm_parts/direct_path.py
  -> subscription_llm_parts/post_layers.py
  -> черновик-заметка в AMO

Telegram Bot API
  -> scripts/run_telegram_ai_agent.py
  -> build_pilot_context_payload()
  -> SubscriptionLlmDraftProvider.build_draft()
  -> тот же direct_path + post_layers
  -> ответ клиенту только при route=bot_answer_self_for_pilot и gate=pass
```

`src/mango_mvp/channels/pilot_profile_runtime.py::current_draft_path()` возвращает
только `direct_path`. Удалённые `dialogue_contract_pipeline.py`,
`rules_engine.py`, `answer_quality_rewriter.py` и humanity-цепочка не являются
запасным живым маршрутом.

## Ответственность модулей

- `subscription_llm_parts/provider.py` собирает контекст и управляет построением
  результата.
- `subscription_llm_parts/direct_path.py` выбирает подтверждённые факты и строит
  основной черновик.
- `subscription_llm_parts/post_layers.py` применяет проверки и допустимые
  преобразования после генерации.
- `output_verification_floor.py` содержит общий детерминированный защитный пол,
  переиспользуемый живым путём: P0-pre-gate, проверку выхода, защиту от
  служебного текста и повторов. Не вся бизнес-логика находится в этом файле.
- `p0_recall_spec.py` содержит каноническую классификацию P0.
- `fact_venue_scope.py` и `fact_scope_spec.py` ограничивают площадку и область
  фактов.
- `subscription_llm_parts/policy_routing.py` допускает, чтобы уверенно
  определённые моделью обычные намерение и тема уточняли план ответа без
  предварительного совпадения со словарём только при явном экспериментальном
  `TELEGRAM_INTENT_MODEL_LED=1`. Обычный профиль сохраняет прежнее поведение до
  парного экзамена M1. При низкой уверенности план не меняется; P0 и остальные
  защитные полы этим правилом не обходятся. Текущая дата управляет отбором
  действующих фактов и передаётся модели и судье; завершённое событие нельзя
  выдавать за текущее или отправлять проверять наличие мест на него. Уверенная
  внешняя тема получает безопасный ответ без выдуманных внешних сведений и без
  повышения маршрута.
- `customer_timeline/bot_safe_runtime_context.py` допускает в контекст только
  разрешённую bot-safe память.
- `apply_payment_confirmation_guard()` и `apply_unstated_subject_guard()`
  вызываются в `provider.py` перед общим `apply_authoritative_output_gate()`.
  Они включаются только явным `TELEGRAM_PAYMENT_SUBJECT_GUARDS=1`; по умолчанию
  и в профиле `pilot_gold_v1` флаг выключен до M1-приёмки и решения владельца.

## Несжимаемые границы

1. Реальный возврат, спор оплаты, серьёзная жалоба и юридический вопрос не
   обходят P0-маршрут.
2. Активный бренд задаётся каналом; подтверждённый чужой бренд исключается.
3. Цена, дата, адрес, расписание, место, бронь и условия утверждаются только из
   подтверждённой базы знаний или слов клиента.
4. ПДн и внутренние метки не выводятся клиенту.
5. `manager_only` не повышается последующим слоем.
6. Wappi-контур создаёт только менеджерский черновик и не отправляет ответ
   клиенту.
7. Telegram-транспорт не классифицирует смысл повторно и не отправляет клиенту
   `manager_only`, `blocked`, ошибку провайдера или текст, не прошедший gate.

Текущая база знаний:

```text
product_data/knowledge_base/kb_release_20260813_v6_8_owner_approved/
  kb_release_v3_snapshot.json
```

Graphify помогает найти связи, но не является источником правды. Любое важное
утверждение о P0, бренде, фактах, ПДн или live-пути проверяется в исходниках и
фактическом runtime.
