# ТЗ-21 / Поставка 1 — персоны автономии

Дата: 2026-06-13

## Артефакты

- Сборщик: `scripts/build_autonomy_personas.py`
- Пакет персон: `product_data/telegram_dynamic_test_sets/autonomy_personas_unpk_20260613.jsonl`
- Тесты: `tests/test_build_autonomy_personas.py`
- Защита старого text-judge/client prompt: `scripts/run_telegram_dynamic_client_sim.py`

## Источники

- Telegram УНПК: `TP UNPK DataExport_2026-05-21/result.json`
- Голос/возражения: `product_data/gold_candidates_paid_proxy_after_calls_tropina_kozlova_20260521_v3_rebuilt_current_runtime/transcripts_all_strong/`

Сырьё не добавлялось в git. В пакет попали только синтетические перефразы, обобщённые objection patterns и буквенные псевдонимы доноров.

## Пакет

Строк: 14:

- 1 `simulator_spec`
- 1 `judge_spec`
- 12 `persona`

Распределение `autonomy_category`:

- `camp`: 6
- `pricing`: 1
- `trial`: 1
- `schedule`: 1
- `format`: 1
- `program`: 1
- `p0`: 1

Распределение `expected_action`:

- `answer_only`: 7
- `capture_lead`: 3
- `book_trial`: 1
- `handoff_manager`: 1

`expected_action` считается правилом из `deal_card.preconditions`, не ручной разметкой.

## Обезличивание

Включены проверки:

- телефоны, email, handles, URL;
- ФИО;
- школы/лицеи/гимназии;
- редкие адреса и площадки;
- точные суммы;
- точные даты;
- дословные фрагменты из тестовых raw fixtures.
- телефоноподобные цифровые последовательности в публичных donor id.

Лимит доноров на персону: 2 (`Telegram seed` + `voice pattern`).

Brand gate: все персоны `brand=unpk`; Фотон-персоны из УНПК-источников не строились.

## Изменение симулятора

Полная persona остаётся в `dynamic_dialog_transcripts.jsonl` для регрейда, но `build_client_prompt()` и текущий `build_judge_prompt()` получают очищенную persona без:

- `expected_action`
- `deal_card`
- `action_*`
- `source_provenance`
- `privacy`
- raw/provenance служебных полей

Причина: Поставка 1 должна не менять текущего text-judge и не давать клиент-модели будущую action-разметку.

## Проверки

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_build_autonomy_personas.py tests/test_telegram_dynamic_client_sim.py tests/test_dialogue_*.py
```

Результат: `433 passed, 1 warning in 4.75s`.

Warning внешний: `urllib3` сообщает, что системный Python собран с LibreSSL; к изменениям ТЗ-21 не относится.

Дополнительная инспекция пакета:

- `safety_violations=[]`
- grep по известным raw-маркерам: совпадений нет
- бренды: только `unpk`
- публичные donor id: только буквы, без телефоноподобных цифровых последовательностей

## Ограничения

- Eval-прогон не запускался: по ТЗ Кодекс делает только pytest, M1-прогон отдельным шагом.
- Action-judge не реализован: Поставка 0 показала отсутствие явного action signal.
- P0-персона использует первый P0-сигнал из Telegram-чата как seed, а не обязательно самое первое сообщение диалога; это зафиксировано как осознанное покрытие hard-P0, без raw-текста в пакете.
- Персоны rule-based и безопасно обобщены; это снижает риск ПДн, но может частично сгладить индивидуальный голос. Нужен регрейд по сырью после M1-прогона.
