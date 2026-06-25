# intent_model_led, 2026-06-25

Ветка: `codex/intent-model-led`

Worktree: `/Users/dmitrijfabarisov/Projects/Mango_intent_model_led`

База: `codex/release-venue-autonomy` / `4caa5eb`

ТЗ: `/Users/dmitrijfabarisov/Claude Projects/Foton/2026-06-25_TZ_intent_model_led_dlya_Kodeksa.md`

## Что подтверждено read-only

- Корень проблемы находится в substring/stem-детекторах: `conversation_intent_plan.py`, `policy_routing.py`, `dialogue_memory.py`.
- Живой direct-path получает план интента из контекста, а route/понижение могут повторно сработать в `apply_conversation_intent_plan_guard` и `apply_autonomy_matrix_guard`.
- Существующий безопасный образец флага взят из `P0_MODEL_LED`: default OFF, явное включение через env/context.

## Что изменено

- Добавлен флаг `TELEGRAM_INTENT_MODEL_LED`, default OFF; в `pilot_gold_v1` не добавлен.
- Direct-path prompt при включенном флаге просит модель вернуть один структурированный сигнал:
  `model_intent = {primary_intent, scope, sense, confidence, reason}`.
- Поддержаны 5 целевых классов: `live_availability`, `schedule`, `address`, `camp`, `price_fix`.
- `model_intent` парсится в `metadata["direct_path_model_intent"]`.
- Все новые решения в route/guard используют один общий сигнал из metadata, а не повторный substring-match.
- Для `live_availability`: настоящий вопрос про наличие мест остаётся менеджерским; ложное `место=площадка/логистика` может быть снято модельным сигналом.
- Runtime summary/selfcheck видит `intent_model_led`.
- Симулятор пишет `run_config.key_flags.intent_model_led`.

## Что осталось детерминированным

- P0/возврат/жалоба/юридическое/спор оплаты.
- Бренд-разделение.
- Числовая и фактологическая проверка.
- Подтверждение оплаты по двум источникам.
- Keyword остаётся дешёвым префильтром: модель не получает право сама включить целевой риск-класс без keyword-сигнала.

## Локальная микро-проверка

Подготовлен набор:

`tasks/_done/2026-06-25_intent_model_led_micro_15.jsonl`

Состав: 15 диалогов, включая:

- `место` как площадка/логистика;
- `когда привезти`;
- `где-то рядом`;
- бытовая `смена`;
- `закрепить материал`;
- настоящие вопросы `есть ли места`, `забронировать место`, `сколько мест осталось`;
- schedule/address/camp/price_fix positive cases.

Локально JSONL проверен: `17 rows`, `15 personas`.

## Команды для M1

Микро OFF:

```bash
TELEGRAM_DIRECT_PATH_PILOT_CONFIG=pilot_gold_v1 \
TELEGRAM_P0_MODEL_LED=1 \
TELEGRAM_PROSE_MODEL_LED=1 \
TELEGRAM_FACT_VENUE_SCOPE=1 \
TELEGRAM_AUTONOMY_SCOPE_PRECISION=1 \
TELEGRAM_INTENT_MODEL_LED=0 \
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
python3 scripts/run_telegram_dynamic_client_sim.py \
  --scenarios tasks/_done/2026-06-25_intent_model_led_micro_15.jsonl \
  --parallel 4 \
  --judge-prompt-version v9.1 \
  --out-dir runs/20260625_intent_model_led_micro_OFF
```

Микро ON:

```bash
TELEGRAM_DIRECT_PATH_PILOT_CONFIG=pilot_gold_v1 \
TELEGRAM_P0_MODEL_LED=1 \
TELEGRAM_PROSE_MODEL_LED=1 \
TELEGRAM_FACT_VENUE_SCOPE=1 \
TELEGRAM_AUTONOMY_SCOPE_PRECISION=1 \
TELEGRAM_INTENT_MODEL_LED=1 \
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
python3 scripts/run_telegram_dynamic_client_sim.py \
  --scenarios tasks/_done/2026-06-25_intent_model_led_micro_15.jsonl \
  --parallel 4 \
  --judge-prompt-version v9.1 \
  --out-dir runs/20260625_intent_model_led_micro_ON
```

Полный replay OFF/ON на M1 нужно запускать на принятом `reliable_replay`/Wappi replay. Разница между плечами должна быть только `TELEGRAM_INTENT_MODEL_LED=0/1`; остальные release-флаги одинаковые.

## Тесты

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q -p no:cacheprovider tests/test_subscription_llm_draft_provider.py -k "intent_model_led or direct_path_model_p0 or p0_model_led"
=> 21 passed, 496 deselected
```

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q -p no:cacheprovider tests/test_conversation_intent_plan.py tests/test_subscription_llm_draft_provider.py tests/test_p0_perifraz.py tests/test_telegram_pilot_p0_register.py tests/test_tz121_brand_e.py tests/test_tz121_brand_e_followup_real.py tests/test_retrofit_channel_brand_tags_in_timeline.py
=> 571 passed
```

```text
bash -lc 'shopt -s nullglob; files=(tests/test_*conversation_intent* tests/test_*policy_routing* tests/test_*intent* tests/test_*p0* tests/test_*brand*); PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q -p no:cacheprovider "${files[@]}"'
=> 100 passed
```

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q -p no:cacheprovider tests
=> 3618 passed, 5 skipped, 1 warning
```

Предупреждение: `urllib3 NotOpenSSLWarning` из локального Python/LibreSSL, не связано с изменением.

## ACK

- Live/AMO/Tallanto/stable_runtime не трогались.
- `pilot_gold_v1` не менялся.
- P0/brand/number/payment-confirm код не ослаблялся.
- Вердикт "в прод" не выносится: нужен регрейд Claude #1 по сырью M1 OFF/ON.

