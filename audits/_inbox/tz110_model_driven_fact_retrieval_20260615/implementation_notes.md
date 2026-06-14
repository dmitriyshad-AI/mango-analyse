# TZ-110 model-driven fact retrieval

Дата: 2026-06-15

## Что реализовано

- Добавлены два экспериментальных флага, оба default OFF:
  - `TELEGRAM_RETRIEVER_NEED_SHADOW`
  - `TELEGRAM_RETRIEVER_MODEL_DRIVEN`
- Флаги не входят в `pilot_gold_v1` и не включаются профилем автоматически.
- В существующий вызов LLM-ретривера добавлена возможность вернуть `needed_facts` без третьего LLM-вызова.
- При `TELEGRAM_RETRIEVER_MODEL_DRIVEN=1` из prompt ретривера убирается `required_fact_keys`, чтобы модель выбирала факты по смыслу без категорийного bias.
- Сквозной `required_fact_keys` вне ретривера не менялся: scope guard, autonomy, missing facts и память продолжают жить на старом контуре.
- Сохранены проверки выбранных id:
  - id должен существовать среди кандидатов;
  - бренд/client_safe/свежесть остаются candidate-time фильтром;
  - конфликт scope мягко переводит exact в adjacent и логируется в `scope_demoted_ids`;
  - при ошибке/таймауте/невалидном JSON/пустом выборе сохраняется keyword fallback.
- В `dynamic_summary.json` добавлен машинный блок `fact_retrieval_trace` по каждому ходу:
  - keyword `required_fact_keys`;
  - модельная декларация `model_needed_facts`;
  - сравнение `declaration_comparison`;
  - candidate count;
  - выбранные exact/adjacent id;
  - scope-demoted id;
  - discarded/invalid id;
  - состояние llm_retrieve и fallback reason;
  - режим A/B;
  - route, P0 signal, brand/scope gate verdicts.

## Осторожная трактовка A-only

`TELEGRAM_RETRIEVER_NEED_SHADOW` добавляет декларацию в тот же prompt, поэтому на уровне реальной модели невозможно математически доказать byte-for-byte parity с id-only prompt без второго LLM-вызова. Чтобы снизить риск, prompt явно требует: сначала выбрать `exact_ids/adjacent_ids` как в обычном режиме, затем описать `needed_facts`.

Абсолютная проверка A-only parity остается предметом ночного OFF vs A/B замера. Флаг default OFF.
