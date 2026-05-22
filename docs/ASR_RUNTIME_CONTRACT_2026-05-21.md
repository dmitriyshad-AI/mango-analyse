# ASR Runtime Contract, 2026-05-21

Цель: не повторять сбой, когда ASR UI запускался в окружении без `mlx_whisper`/`gigaam`, а batch массово уходил в `dead`.

## Текущий статус

- Текущий активный ASR runtime: `.venv-asrbench/bin/python`.
- Текущий активный batch: `product_data/mango_update_after_20260512_20260521_v1/asr_ui_batch`.
- Текущая batch DB: `product_data/mango_update_after_20260512_20260521_v1/asr_ui_batch/mango_after_20260512_asr_only.sqlite`.
- `.venv-asrbench` нельзя удалять, пока свежие Mango-звонки распознаются через этот runtime или пока не подготовлена замена.

## Обязательная проверка перед ASR

Перед запуском ASR выполнить:

```bash
cd "/Users/dmitrijfabarisov/Projects/Mango analyse"

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
python3 scripts/check_asr_runtime_contract.py
```

ASR можно запускать только если:

- `active_runtime_ok=true`;
- preferred Python указывает на рабочее окружение;
- окружение видит все модули:
  - `sqlalchemy`
  - `dotenv`
  - `mango_mvp.cli`
  - `mango_mvp.gui`
  - `mlx_whisper`
  - `gigaam`

## Что считать опасным

- Любой launcher, который ссылается на `stable_runtime/venv_stable.broken_20260407/bin/python`, является legacy entrypoint. Этот venv уже удалён, такие скрипты нельзя запускать без обновления.
- Любой launcher, который напрямую ссылается на `stable_runtime/venv_stable/bin/python`, тоже опасен: этот venv сейчас не проходит ASR runtime contract.
- На 2026-05-21 shell-launchers в `stable_runtime` переведены на `.venv-asrbench`; если такие ссылки вернутся, это регрессия.
- Проверка только `sqlalchemy`/`dotenv` недостаточна: UI может открыться, но ASR упадёт позже.

## Чего не делать

- Не удалять `.venv-asrbench` как “benchmark venv”, пока он является рабочим ASR runtime.
- Не чинить старые `stable_runtime/run-ui-*.sh` массовым search/replace во время активного ASR batch.
- Не запускать несколько ASR workers без отдельной причины: Whisper/MLX и GigaAM сами активно используют ресурсы.
- Не переносить и не удалять `stable_runtime/messages35_asr_only_20260506`: это lineage source для canonical master и не мусор.

## Безопасный следующий шаг после завершения текущего ASR

1. Дождаться `transcription_status.done = total_calls`.
2. Убедиться, что `dead_letter_stage={}`.
3. После отдельного подтверждения запускать Resolve+Analyze.
4. После accepted rebuild отдельно решить, какие старые launchers переписать или пометить deprecated.
