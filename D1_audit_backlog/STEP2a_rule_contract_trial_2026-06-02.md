# Контракт правила (Шаг 2a): «пробное». 2026-06-02.

Автор: Клод 1. Источник — `_trial_safe_template` (153 LOC) + `_foton_offline_free_trial_guard_template` +
9 TRIAL-констант (228-265). Прочитано по сырью.

## Как сейчас (лазанья, 153 строки)
Мешает: распознавание пробного (пробн/фрагмент) + подвид (платно/бесплатно, какие данные нужны, как
получить ссылку, «жду фрагмент») + бренд + формат (онлайн-фрагмент vs очное) + слоты класс/предмет +
скрытые исключения (прямой запрос менеджера; «не онлайн»; free-trial-guard).

## Целевой контракт

```yaml
rule_id: trial
title: Пробное занятие / фрагмент занятия
intent: trial_inquiry
intent_subvariants: [online_fragment, offline_trial, process_how, data_needed, fragment_access, payment, ack]
required_fact_keys:
  - trial.foton.online_fragment        # Фотон: онлайн — фрагмент занятия
  - trial.foton.process                # как получить/оформить
  - trial.unpk.*                        # пробное УНПК
data_rules:
  foton_offline_free_trial: not_promised   # НЕ обещать бесплатное ОЧНОЕ пробное (free-trial-guard)
  online_fragment: foton_online          # фрагмент занятия = онлайн-формат Фотона
  format_match: true                     # очное/онлайн пробное — по формату вопроса
data:
  foton: {online: fragment_of_lesson, offline_free_trial: false}
brand_split: true
blocking_conditions:
  - direct_manager_request   # «передайте менеджеру/хочу менеджера» БЕЗ «как/способ/ссылка» → менеджер, не отвечать пробным
  - offline_free_trial_promise # обещание бесплатного очного пробного → block (гейт)
route_effect: bot_answer_self  # если факт есть; иначе уточнение/менеджер
text_effect: composer_generates_from_data
preserve_exceptions:
  - «передайте менеджеру» БЕЗ «как/способ/получить/ссылка/запись» → чистый хендофф, не ответ про пробное
  - «не онлайн / только не онлайн» → не предлагать онлайн-фрагмент
  - free-trial-guard: НЕ обещать бесплатное очное пробное (Фотон)
  - подвид data_needed («какие данные нужны / что прислать») ≠ payment («платно?») ≠ access («как получить»)
```

## Слои
- **Planner:** intent=trial + subvariant + формат + класс/предмет + uncertainty. Заменяет тяжёлую keyword-
  детекцию (пробн/фрагмент/платно/какие данные/как получить).
- **Слой 2:** data + правила (free-trial-guard, online_fragment, format_match).
- **Composer:** ответ из факта под subvariant; 9 текст-констант исчезают.
- **Гейт:** обещание бесплатного очного пробного → block; прямой запрос менеджера → manager.

## Негатив-тесты
1. Фотон «бесплатное очное пробное есть?» → НЕ обещать (free-trial-guard).
2. «передайте менеджеру» (без «как») → чистый хендофф, не ответ про пробное.
3. «не онлайн, очно хочу» → не предлагать онлайн-фрагмент.
4. data_needed «что прислать для фрагмента?» → про данные, не про цену.
5. ПОЗИТИВ: Фотон «можно пробное онлайн?» → фрагмент занятия из факта, route=answer_self.
