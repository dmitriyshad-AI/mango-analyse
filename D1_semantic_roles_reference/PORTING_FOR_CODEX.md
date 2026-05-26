# Инструкция переноса для Кодекса

Эта папка — РЕФЕРЕНС. Claude (Cowork) не правил файлы репозитория. Здесь лежит рабочая
логика + тесты. Перенос делает Кодекс на отдельной ветке, затем меряет round-5.

## Шаг 0. Куда положить модули

- `reference/semantic_roles.py` → `src/mango_mvp/channels/semantic_roles.py`
- `reference/decision_policy.py` → влить в `conversation_intent_plan.py` и
  `answer_safety_classifier.py` (см. ниже), либо положить рядом как
  `src/mango_mvp/channels/answer_plan.py` и вызывать из плана.

ВАЖНО при переносе: в `semantic_roles.py` функция `has_marker` сейчас ДУБЛИРУЕТ
`text_signals.has_marker` (чтобы референс был автономным). При переносе УДАЛИ дубль и
импортируй:
```python
from mango_mvp.channels.text_signals import has_marker, has_any_marker
```
Это и есть «не плодить подстрочные проверки».

## Шаг 1. Подключить роли в план диалога

В `conversation_intent_plan.build_conversation_intent_plan` вызвать
`tag_message_roles(current_message)` ОДИН раз и прокинуть результат в:
- `ConversationIntentPlan` новыми полями: `topic_roles: tuple[str,...]`,
  `payment_method: str`, `payment_source: str`, `refund_frame: str`,
  `enrollment_vs_recording: str`, `transfer_sense: str`;
- `to_prompt_view()` — добавить эти поля, чтобы их видела генерация.

## Шаг 2. ЗАМЕНИТЬ старый каскад (это главное, не «добавить»)

В `conversation_intent_plan._fact_scope_constraints` (строки ~429–507) сейчас длинная
цепочка частных `if _has_any_marker(...)`. Её нужно ВЫТЕСНИТЬ: scope выводить из ролей
(`training_format`, `payment_method`, `payment_source`, `enrollment_vs_recording` и т.п.),
а не из набора подстрочных проверок. Конкретно убрать/свернуть ветки:
- `trial_offline` / `trial_online_fragment` → из `training_format`;
- `dolyami_parts` / `installment_bank` → из `payment_method`;
- `matkap_process` / `tax_deduction` → из `payment_source`;
- `offline_recordings` / `online_recordings` → из `enrollment_vs_recording=recording`
  + `training_format`;
- разведение `enroll` vs `recording` для слова «запись» — из `enrollment_vs_recording`
  (удалить локальные `_asks_enrollment_signup` / `_asks_lesson_recording`, если их роль
  полностью покрыта).

Приёмка анти-«крот»: число `if`-веток в `_fact_scope_constraints` после переноса
заметно меньше; старые ветки удалены, не закомментированы.

## Шаг 3. Заменить refund-regex типизацией

В `p0_recall_spec.py` сейчас `BENIGN_HYPOTHETICAL_REFUND_RE`,
`PRESALE_REFUND_POLICY_RE`, `PRESALE_REFUND_CONTEXT_RE` и `_is_benign_hypothetical_refund`
ловят предпродажный возврат узкими шаблонами (именно поэтому «передумаю до начала» не
поймался). Заменить: `is_benign_hypothetical_refund(text)` должна возвращать
`tag_message_roles(text).refund_frame == "presale_policy"`.

В `answer_safety_classifier.classify_answer_safety`:
- `refund_frame == "dispute"` → как сейчас P0 refund;
- `refund_frame == "presale_policy"` → НЕ P0 (semantic_non_p0 ветка), бот отвечает;
- внешний P0 (legal/complaint/payment_dispute, `p0_latch`) НЕ ослаблять — как сейчас.

Приёмка: три узких refund-regex удалены (или сведены к одному источнику через роли),
а не оставлены рядом «на всякий случай».

## Шаг 4. Политика и шаблон-fallback

Использовать `decision_policy.build_answer_plan`:
- `answer_topics` прокинуть в промпт генерации (ответить на все темы);
- `forbidden_pairs` прокинуть в промпт и в детерминированный гейт
  (`matkap+installment` → не предлагать рассрочку вместе с маткапиталом);
- `template_allowed=False` при наличии содержательного ответа — соответствует уже
  заявленному «класс B: неприкосновенность более точного ответа». Проверить, что
  `answer_quality_rewriter` и шаблонная подстановка уважают этот флаг (не перезаписывают).

## Шаг 5. Тесты

Перенести `tests/test_semantic_roles.py` в набор репозитория (поправить импорт на
`mango_mvp.channels.semantic_roles` / `...answer_plan`). Прогнать. Затем — round-5 на
свежем holdout, приёмка в `TZ_semantic_roles_layer_2026-05-25.md` §7.

## Чего НЕ делать

- Не оставлять старые ветки «спящими» — это превратит «замену» в «наращивание».
- Не переносить дубль `has_marker` — импортировать из `text_signals`.
- Не трогать бренд-разделение, P0-консервативность, политику «ты бот?».
