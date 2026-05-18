# Инструкция Codex'у — итерация v3.2
Дата: 2026-05-18
От: Claude (после ревью v3.1 и решений Дмитрия)

---

## 1. Что Claude уже сделал (взять из handoff'а)

В `claude_to_codex_v3_handoff_2026-05-17/` обновлены 4 YAML. Изменения:

### `facts_for_bot_FOTON.yaml` и `facts_for_bot_UNPK.yaml`
- Раздел `promo_codes` → `status: removed_from_bot`. Бот больше не работает с промокодами.
- `bot_behavior.when_client_asks_about_promo` → handoff менеджеру с фразой «Спасибо, передам менеджеру — он подскажет актуальные акции и промокоды».

### `facts_internal_only.yaml`
- `teacher_promo_codes` → `status: archived_2026_05_18`, `bot_use: false`. Промокоды преподавателей хранятся как справка для менеджера, не для бота.
- `open_discrepancies.telegram_handle_foton` → `status: closed`. Решение Дмитрия 2026-05-18: `@unpkmfti` оставляем как есть. В будущем маркетинг поменяет.

### `bot_policy.yaml`
- Новый раздел `promo_codes_policy` — единое правило для двух ботов: `route: draft_for_manager`, не работаем с промокодами.

---

## 2. Решения Дмитрия 2026-05-18 (для контекста)

1. **`@unpkmfti` оставляем.** В будущем маркетинг поменяет, но сейчас не блокер.
2. **Промокоды убираем из бота полностью.** Слишком много без `valid_until`, риск истёкших промокодов.
3. **`document_verified` факты использовать сразу в `bot_answer_self_for_pilot`.** Не требовать дополнительного `rop_approved` для пилота. Claude в прошлом отчёте перестраховался — это снято.

---

## 3. Что нужно от Codex (по приоритету)

### P0 — без этого пилот не запускать

**3.1. Перегенерировать v3-snapshot с актуальными YAML**

В Claude handoff обновлены 4 YAML. Запустить `scripts/build_kb_release_v3_from_claude_handoff.py` ещё раз, чтобы:
- из `facts_registry` исчезли промокоды как клиентские факты (22 фактов уйдут в архив)
- появилось правило `promo_codes_policy` в `bot_policy`
- решение по `@unpkmfti` отразилось в quality report как закрытое

**3.2. Инвариант в `semantic_pass`**

Добавить правило: если `requires_manager_confirmation=true`, то `freshness_status` НЕ может быть `verified`/`document_verified`. Должен быть `needs_owner_confirmation` или `dynamic_needs_check`.

Это закроет 3 факта с логическим конфликтом, которые я нашёл в v3.1. После выкидывания промокодов их остаётся 2 (расписание занятий, места в группах ЛВШ) — для них правильный статус `dynamic_needs_check`.

**3.3. 5 новых правил `semantic_pass`** (из моего предыдущего ответа)

1. **Срок действия**: для `verified` факта обязателен `valid_until` или `freshness_check_date`.
2. **Конфликт статусов**: `requires_manager_confirmation=true` + `allowed_for_client_answer=true` — блокирующая ошибка (см. 3.2).
3. **Непустой client_safe_text**: если `allowed_for_client_answer=true`, текст обязан быть непустым.
4. **Cross-brand в manager-полях**: проверять `manager_check_text` на упоминания юр.лиц чужого бренда (хотя бы предупреждение, не блокировка).
5. **Согласованность product_key**: два разных продукта в YAML не могут дать один и тот же `product_key`.

**3.4. Финальный микрофикс «110 ₽»**

В v3.1 остался 1 факт `pair_duration_minutes=110` развёрнутый как «110 ₽». Поле `pair_duration_minutes` должно идти в `course_parameter`, не `price`. Это и есть пример правила 3.3.5.

**3.5. 50 smoke-вопросов в JSONL**

Файл со списком: `kb_release_v2_claude_layer_2026-05-17/codex_v3_final_review/SA6_smoke_questions_50.md`.

25 FOTON + 25 UNPK. Каждый имеет `client_message`, `expected_route`, `expected_in_draft`, `forbidden_in_draft`. Конвертировать в JSONL по структуре существующих `stage6_fixtures_*.jsonl`, прогнать через real `codex exec`.

Целевые метрики:
- `brand_separation_violation = 0`
- `unsupported_numeric_promises = 0`
- `expected_route_hit = 100%` для P0
- `became_more_substantive ≥ 15/20` на каждый бренд

### P1 — желательно перед пилотом на клиентах

**3.6. Переделать `rop_question` в `approval_queue`**

Сейчас 8-10 шаблонных формулировок на 634 строки. РОП будет проходить 4-6 часов вместо целевых 3-4.

Шаблон конкретного вопроса: «Подтвердить, что [fact_text] актуально и можно использовать в клиентском ответе?» Подставлять `fact_text` динамически. Это сократит время ревью РОПа в 2 раза.

**3.7. Зачистка остатков**

- 7 фактов с пустым `client_safe_text` при `allowed=true` — либо заполнить, либо понизить статус.
- 3 факта с маркетинговыми обещаниями («лучшие преподаватели страны») без `manager_check_text` — добавить.
- 2 факта с упоминанием «АНО ДПО» в `manager_check_text` — допустимо, но стоит проверить, что менеджер видит в draft (а не в служебных полях).

### P2 — после пилота

**3.8. Регрессионные тесты**

Каждый баг, который я находил за все циклы (v1, v2, v3.0, v3.1) — превратить в отдельный регрессионный тест в `tests/test_kb_semantic_regressions.py`. Например:
- `test_no_35_rub_per_year_for_course_parameter`
- `test_no_split_ranges_without_pair_id`
- `test_no_two_products_under_same_key`
- `test_no_forbidden_to_say_in_client_safe_text`
- `test_no_license_number_in_allowed_text`

10-15 тестов. Каждый — короткий, на 1 баг. Это страховка от повторения.

---

## 4. Что Codex НЕ должен делать

- **Не требовать `rop_approved` для `document_verified` фактов в пилотном режиме.** Решение Дмитрия 2026-05-17: `bot_answer_self_for_pilot` разрешён для всех verified фактов из утверждённых источников. Дополнительная RОП-метка — для production, не для пилота.
- **Не возвращать промокоды в client_safe_text.** Если факт пришёл из старой версии — он `archived_2026_05_18`.
- **Не догадываться по `@unpkmfti`.** Решение зафиксировано — оставить.
- **Не переписывать `brand_rules.yaml` и общую структуру `bot_policy.yaml`.** Они синхронизированы.

---

## 5. Контрольный список для возврата

Перед тем как Codex отдаст мне следующий handoff, проверить:

- [ ] `facts_registry` не содержит промокодов как клиентских фактов (проверка: `grep promo client_safe_text` → 0 строк)
- [ ] 0 фактов с `requires_manager_confirmation=true` + `freshness_status=verified` одновременно
- [ ] 0 фактов «N ₽» где N < 1000 (фильтр на course_parameter)
- [ ] 100% facts с `source_id`, который есть в `source_registry`
- [ ] `rop_question` имеет 50+ уникальных формулировок (а не 8-10 шаблонов)
- [ ] Smoke-тест на 50 вопросах прогнан, метрики ОК
- [ ] `claude_handoff_response.md` обновлён с фактическими цифрами

Если все ☑ — handoff готов, можно запускать пилот на сотрудниках.

---

## 6. Что произойдёт после следующего handoff'а

1. Я пройдусь 6 субагентами по той же схеме (атомарность, brand safety, source registry, approval queue, ответы команды, snapshot).
2. Дополнительно прогоню semantic-проверки на smoke-результатах (искал ли бот реально цены? не сказал ли что-то странное?).
3. Если 0 блокирующих — пилот на 3-5 сотрудниках на 3 дня.
4. Если в пилоте на сотрудниках 0 жалоб — пилот на 1-2 лояльных клиентах.

---

## 7. Краткая хронология (для контекста)

| Дата | Что | Результат |
|---|---|---|
| 17.05 утро | Claude собирает 5 YAML из 11 источников | Базовая структура готова |
| 17.05 день | 10 субагентов обогащают из Codex pack | Цены, лицензии, продукты — все из реальных документов |
| 17.05 вечер | Ответы команды на 11 вопросов | q3, q7-q13, темы 11/12/17 закрыты |
| 17.05 ночь | Codex выпустил v2 | 90% правильно, 4 критических бага |
| 17.05 поздно | Codex выпустил v3.0 | 4 бага из 4 починены, 7 жёлтых, 12 новых семантических ошибок |
| 18.05 утро | Codex выпустил v3.1 + semantic_pass | 6 из 7 жёлтых починены, semantic_pass работает |
| 18.05 день | Дмитрий принял 3 решения | Промокоды убираем, @unpkmfti оставляем, rop_approved не нужен в пилоте |
| 18.05 — это сейчас | Claude обновил 4 YAML под решения | Ждём Codex v3.2 |
