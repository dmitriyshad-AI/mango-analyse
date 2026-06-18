# TZ-147 Report — углубление P0-детектора и независимый выходной передерив

Дата: 2026-06-18
Ветка: `codex/tz147-p0-deep-output-carry`
База: `a079e914`

## Что сделано

1. Добавлен флаг `TELEGRAM_P0_DEEP_MATCH`, default OFF, env-only.
   - Для матчинга P0 снимается только внутрисловный дефис между кириллическими буквами: `спи-сали` -> `списали`.
   - Для класса `payment_dispute` добавлена склейка только соседних предложений.
   - Добавлены платёжные P0-синонимы: активация, подключение, не выслали, не назначили, не активировали, заблокирован.
   - Новые слова срабатывают только в паре с фактом оплаты/списания/платежа.

2. Добавлен флаг `TELEGRAM_P0_OUTPUT_MODEL_CARRY`, default OFF.
   - Выходной гейт теперь умеет, при включенном флаге, поднимать `hard_p0` из `metadata["direct_path_model_p0"].is_p0`.
   - Дополнительного вызова модели нет.
   - Модельный сигнал только усиливает блокировку; он не может разблокировать детерминированный P0-пол.

3. Добавлен M1-набор:
   - `product_data/telegram_dynamic_test_sets/p0_deep_match_tz147_20260618.jsonl`
   - 28 диалогов: 15 POS и 13 NEG.
   - Цель: сравнить OFF/ON на классе `paid_no_access` и анти-передоловах.

## Проверки

Точечная проверка:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_p0_perifraz.py tests/test_answer_safety_classifier.py tests/test_subscription_llm_draft_provider.py -k 'p0 or authoritative_output_gate'
```

Результат:

```text
166 passed, 460 deselected in 2.56s
```

Полная проверка:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q
```

Результат:

```text
3383 passed, 5 skipped, 1 warning in 52.28s
```

Предупреждение: локальный `urllib3 NotOpenSSLWarning` из-за LibreSSL в системном Python. К задаче не относится.

## Самопроверка безопасности

- Оба новых флага выключены по умолчанию.
- `TELEGRAM_P0_DEEP_MATCH` не расширяет общий gap старого `PAYMENT_DISPUTE_RE`.
- Кросс-предложенческая проверка ограничена соседними предложениями.
- Нормализация применяется к копии текста для матчинга; evidence и клиентский текст не переписываются.
- `TELEGRAM_P0_OUTPUT_MODEL_CARRY` читает уже существующий модельный сигнал и не делает новый LLM-вызов.
- Сбой модели не ослабляет детерминированный P0: hard-regex остаётся отдельным pre-gate.

## Что не делалось

- Флаги не добавлялись в `pilot_gold_v1`.
- M1-прогон не запускался локально.
- AMO/Tallanto/CRM/stable_runtime не трогались.
- ASR/R+A не запускались.

## M1 bundle

Бандл нужно собрать после коммита, чтобы `manifest.json` указывал на финальный commit.

Ожидаемые флаги для ON-прогона:

```bash
export TELEGRAM_P0_DEEP_MATCH=1
export TELEGRAM_P0_OUTPUT_MODEL_CARRY=1
```

OFF-прогон: оба флага не выставлять.

Набор:

```text
product_data/telegram_dynamic_test_sets/p0_deep_match_tz147_20260618.jsonl
```

Снапшот:

```text
product_data/knowledge_base/kb_release_20260617_v6_7_staging_r4_1/kb_release_v3_snapshot.json
```

Если путь снапшота отличается в окружении M1, использовать актуальный `v6_7_staging_r4_1`.

## Остаточные риски

1. Поведенческий эффект пока подтверждён тестами, но не M1-транскриптами.
2. `TELEGRAM_P0_OUTPUT_MODEL_CARRY` зависит от качества уже существующего `direct_path_model_p0`; поэтому включать только после регрейда транскриптов.
3. Более широкие платёжные формулировки могут увеличить менеджерские маршруты, если реальные клиенты часто обсуждают оплату и доступ гипотетически. Для этого добавлены NEG-кейсы, но финальное решение — по M1.
