# M1 sets: closing-fix tz142 + tz147 deep-match harness

Дата: 2026-06-27
Ветка: `main`

## Что сделано

- Добавлен новый набор `product_data/telegram_dynamic_test_sets/closing_fix_tz142_20260627.jsonl`.
- Восстановлен существующий набор `product_data/telegram_dynamic_test_sets/p0_deep_match_tz147_20260618.jsonl` из ветки `codex/tz147-p0-deep-output-carry`; заново не придумывался.
- Добавлен детерминированный харнесс `scripts/check_p0_deep_match.py`.

## Closing-fix набор

Состав: 20 persona-сценариев.

- `pos_closing`: 8, Фотон 4 / УНПК 4.
- `neg_p0_on_closing`: 4.
- `neg_fabrication_on_closing`: 3.
- `neg_brand_on_closing`: 3.
- `neg_over_handoff_on_closing`: 2.

Каждый сценарий заканчивается фиксированной закрывающей репликой в последнем `behaviors[]`. Набор предназначен для ON-only прогона на M1 с `TELEGRAM_TONE_CLOSE_DETECT=1`, `--parallel 4`, судья v9.1, один прогон. OFF не нужен: слой `apply_tone_close_detect_layer` при выключенном флаге является no-op.

Перед M1-прогоном набор должен пройти semantic-safety ревью Claude #1: бренд-разделение, отсутствие выдуманных фактов, корректные P0-ожидания.

## tz147 deep-match

Набор: `p0_deep_match_tz147_20260618.jsonl`, 28 persona-строк.

- POS: 15.
- NEG: 13.

Харнесс вызывает `codes_from_text()` в двух режимах:

- OFF: `TELEGRAM_P0_DEEP_MATCH` снят.
- ON: `TELEGRAM_P0_DEEP_MATCH=1`.

В текущем `main` код `TELEGRAM_P0_DEEP_MATCH` ещё не портирован, поэтому харнесс предупреждает, что OFF/ON могут быть одинаковыми. Это ожидаемо: задача переносит измерительный инструмент, а не включает/портирует сам `tz147`-код.

## Проверки

Команды:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m py_compile scripts/check_p0_deep_match.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/check_p0_deep_match.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 - <<'PY'
from pathlib import Path
from scripts.run_telegram_dynamic_client_sim import load_dynamic_sim_input
sim = load_dynamic_sim_input(Path('product_data/telegram_dynamic_test_sets/closing_fix_tz142_20260627.jsonl'))
print('personas', len(sim.personas))
print('judge', sim.judge_spec.get('title'))
print('simulator', sim.simulator_spec.get('title'))
PY
```

Ожидание сейчас: скрипт отрабатывает и честно показывает текущий baseline. После порта `tz147`-кода можно запускать:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/check_p0_deep_match.py --strict
```

## Границы

- Флаги не включались.
- M1-прогоны не запускались.
- AMO/Tallanto/CRM/client writes = 0.
- `tz147 carry` не измерялся: это отдельная задача с M1 и двумя плечами.
