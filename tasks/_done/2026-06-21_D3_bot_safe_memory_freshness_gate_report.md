# D3: гейт свежести bot-safe памяти

Дата: 2026-06-21
Ветка: `codex/d3-phase01-botsafe-integration`
Флаг: `TELEGRAM_BOT_SAFE_CRM_CONTEXT`, default OFF
Статус: реализация готова на ветке, в `main` не влита.

## Что изменено

Проброшен безопасный статус следующего шага из bot-safe выжимки в прямой путь бота.

Изменения по коду:

- `src/mango_mvp/customer_timeline/read_api.py`
  - `project_bot_context()` теперь отдаёт `next_step_status`, если он есть в `metadata.next_step.status`;
  - отдаётся только статус из списка `active`, `needs_manager_review`, `empty`;
  - текст шага из `metadata.next_step.display_text/action` не отдаётся.
- `src/mango_mvp/customer_timeline/bot_safe_runtime_context.py`
  - `_safe_items_for_brand()` добавляет `next_step_status` в items измерительного/runtime-контура;
  - поддерживает как уже спроецированное поле `next_step_status`, так и fallback из `metadata.next_step.status`.
- `src/mango_mvp/channels/subscription_llm_parts/direct_path.py`
  - `_direct_path_bot_safe_context_items()` сохраняет `next_step_status`;
  - `_direct_path_bot_safe_context_prompt_block()` добавляет инструкцию:
    - `active` -> продолжать нить и называть шаг без лишних оговорок;
    - `needs_manager_review` / `empty` -> шаг не подтверждён, не утверждать клиенту, предложить уточнить с менеджером;
    - датированную историю с неподтверждённым шагом подавать как прежние заметки: «по прежним заметкам, актуальность уточню»;
  - trace теперь включает `next_step_statuses`, чтобы в измерительном контуре было видно, что статус дошёл.

## Что принципиально не менялось

- Флаг остался default OFF.
- Live-бот, AMO, Tallanto и боевая БД не трогались.
- Цены, даты и условия по-прежнему можно брать только из блока «Факты по вашему вопросу».
- PII-скан bot-safe текста не ослаблялся.
- Бренд-фильтр active brand + unknown / исключение чужого бренда не менялся.
- `customer_profile` целиком в бот не передаётся.

## Проверка NEG

### Active без лишнего хеджа

Контрольный `active_only` prompt block:

```text
Если статус следующего шага «active», продолжай эту нить и называй шаг без лишних оговорок.
1. Фотон: клиент уже спрашивал про онлайн-курс. Следующий шаг: отправить расписание. (2026-06-21) [статус следующего шага: active]
```

В active-only блоке отсутствует фраза:

```text
по прежним заметкам, актуальность уточню
```

### Needs manager review получает маркер свежести

Контрольный смешанный prompt block содержит:

```text
Если статус следующего шага «needs_manager_review» или «empty», следующий шаг НЕ подтверждён: не утверждай его клиенту, предложи уточнить с менеджером. Датированную историю с таким статусом подавай как прежние заметки: «по прежним заметкам, актуальность уточню».
2. Без бренда: клиент ранее уточнял удобный формат. (2026-06-20) [статус следующего шага: needs_manager_review]
```

### Статус дошёл до измерительного контура

Тесты подтверждают:

- `build_bot_safe_crm_context()` возвращает items с `next_step_status`;
- `build_bot_prompt_context()` в динамическом симуляторе получает `next_step_status`;
- `run_amo_wappi_draft_loop` получает `next_step_status`;
- trace прямого пути содержит `next_step_statuses`.

## Тесты

Точечный набор:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q \
  tests/test_bot_safe_runtime_context.py \
  tests/test_bot_safe_direct_path_context.py \
  tests/test_customer_timeline_read_api.py \
  tests/test_telegram_dynamic_client_sim.py::test_dynamic_context_can_inject_bot_safe_summary_by_customer_id \
  tests/test_run_amo_wappi_draft_loop.py::test_context_builder_injects_only_bot_safe_crm_context_when_enabled
```

Результат:

```text
19 passed in 1.07s
```

Полный набор:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q
```

Результат:

```text
3515 passed, 5 skipped, 1 warning in 58.34s
```

Warning: системный `urllib3 NotOpenSSLWarning`, не связан с изменением.

## Остаточный риск

Это prompt-level гейт, а не детерминированный переписчик ответа. Финальную эффективность нужно мерить D7 OFF/ON на 14/16/18 и active-кейсах 03/10/11/17 по реальным черновикам.
