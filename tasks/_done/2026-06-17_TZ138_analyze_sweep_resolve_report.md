# ТЗ-138: Analyze 3 модели + Resolve 25 звонков

Дата: 2026-06-17  
Ветка: `codex/tz138-analyze-sweep`  
Worktree: `/Users/dmitrijfabarisov/Projects/Mango_tz138_analyze_sweep`  
Исходное ТЗ: `/Users/dmitrijfabarisov/Projects/Mango analyse/tasks/_inbox_codex/2026-06-17_TZ138_analyze_svip_3_modeli_i_resolve_test.md`

## Итог

Часть A выполнена: одни и те же 100 `canonical_calls.id` прогнаны через `scripts/run_analyze_ab_test.py` на 3 конфигурациях:

| Конфигурация | Модель | Режим рассуждения | Статус | Время |
|---|---:|---:|---:|---:|
| `mini_high` | `gpt-5.4-mini` | `high` | 100/100 done | 4057.698 сек / 67.63 мин |
| `gpt54_medium` | `gpt-5.4` | `medium` | 100/100 done | 2244.777 сек / 37.41 мин |
| `gpt55_medium` | `gpt-5.5` | `medium` | 100/100 done | 1905.170 сек / 31.75 мин |

Часть B выполнена: Resolve сохранён для 25 звонков из той же сотни. CLI сообщил `processed=25`, `success=25`, `failed=0`, `manual=0`, `llm_used=0`, `rescue_used=0`. На выбранном наборе Resolve прошёл локальным rule/stereo путём, без вызовов модели, хотя `codex_cli` был включён.

Каноническая база не импортировалась и не менялась. `run_analyze_ab_test.py` работал через копии БД в `--out-dir`; Resolve также запускался на отдельной копии БД.

## Артефакты

Корневая папка артефактов, игнорируется git:

`/Users/dmitrijfabarisov/Projects/Mango_tz138_analyze_sweep/product_data/customer_profiles/tz138_analyze_resolve_20260617`

Основные файлы:

- `ids_100.txt` — общий список 100 id.
- `resolve_ids_25.txt` — список 25 id для Resolve.
- `sample_manifest.csv` — сводка выборки.
- `sample_summary.json` — агрегаты по выборке.
- `run_manifest.json` — машинный манифест прогонов.
- `commands.md` — команды запуска.

Analyze JSON:

- `analysis_json/mini_high/mini_high_analysis_json.jsonl` — 100 строк.
- `analysis_json/gpt54_medium/gpt54_medium_analysis_json.jsonl` — 100 строк.
- `analysis_json/gpt55_medium/gpt55_medium_analysis_json.jsonl` — 100 строк.
- Также рядом лежат per-id `*.analysis.json`.

Resolve:

- `resolve/resolve_outputs/manifest.json`.
- `resolve/resolve_outputs/<seq>_<id>/resolve.json`.
- `resolve/resolve_outputs/<seq>_<id>/transcript_text.txt`.

Важно: Resolve-артефакты содержат исходные тексты звонков и не должны попадать в git.

## Выборка

Всего id: 100.  
Из blacklist-77 в сотне: 15.  
Для Resolve: 25 id, из них blacklist: 9.

Распределение 100 id по длительности:

- `short_lt120`: 23
- `medium_120_360`: 24
- `long_360_900`: 25
- `very_long_900_1800`: 21
- `huge_ge1800`: 7

Оценочное распределение по бренду:

- `foton`: 35
- `unpk`: 35
- `mixed`: 8
- `unknown`: 22

Список 100 id:

```text
16628, 63275, 53079, 57873, 12617, 62170, 19055, 63178, 58340, 64128,
61646, 62505, 62654, 15112, 19871, 25846, 63679, 2426, 4765, 5655,
6194, 6332, 6820, 7193, 2706, 3393, 3447, 3580, 3694, 4379,
4455, 530, 1915, 2086, 3436, 3453, 3467, 3605, 14241, 14431,
20056, 21819, 22286, 23222, 3033, 1449, 2470, 4468, 1313, 4,
7, 51, 140, 1, 12, 18, 20, 30, 3, 11,
13, 31, 50, 2727, 4986, 5119, 6459, 42171, 44488, 56190,
22, 27, 37, 131, 179, 199, 220, 10, 26, 28,
63, 90, 113, 115, 123, 2, 8, 15, 16, 23,
36, 39, 40, 933, 1319, 1705, 1946, 2037, 2799, 9185
```

Список 25 id для Resolve:

```text
16628, 63275, 53079, 57873, 12617, 62170, 25846, 63679, 44488, 56190,
22, 27, 37, 131, 179, 199, 19055, 3033, 63178, 1313,
4, 7, 51, 140, 61646
```

## Команды

Общий `ids-file` был один и тот же для всех трёх Analyze-прогонов:

`/Users/dmitrijfabarisov/Projects/Mango_tz138_analyze_sweep/product_data/customer_profiles/tz138_analyze_resolve_20260617/ids_100.txt`

`stable_runtime/run-cli.sh` не использовался, потому что его окружение на этой машине падает на импорте `sqlalchemy`. Runtime не чинился и не менялся. Вместо этого использован `/usr/bin/python3 -m mango_mvp.cli` через параметр `--cli /usr/bin/python3` и `PYTHONPATH=src`.

### mini_high

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src PATH="/Applications/Codex.app/Contents/Resources:$PATH" LLM_CACHE_DIR="/Users/dmitrijfabarisov/Projects/Mango_tz138_analyze_sweep/product_data/customer_profiles/tz138_analyze_resolve_20260617/.llm_cache" python3 scripts/run_analyze_ab_test.py --source-db "/Users/dmitrijfabarisov/Projects/Mango analyse/stable_runtime/canonical_master_20260523_audio_working_store_v1/canonical_calls_master.db" --ids-file product_data/customer_profiles/tz138_analyze_resolve_20260617/ids_100.txt --out-dir product_data/customer_profiles/tz138_analyze_resolve_20260617/analyze/mini_high --cli /usr/bin/python3 --arms mini_high:gpt-5.4-mini:full --reasoning high --prompt-profile full --sample-size 100 --provider codex_cli --timeout-sec 600 --keep-export-files
```

### gpt54_medium

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src PATH="/Applications/Codex.app/Contents/Resources:$PATH" LLM_CACHE_DIR="/Users/dmitrijfabarisov/Projects/Mango_tz138_analyze_sweep/product_data/customer_profiles/tz138_analyze_resolve_20260617/.llm_cache_gpt54_medium" python3 scripts/run_analyze_ab_test.py --source-db "/Users/dmitrijfabarisov/Projects/Mango analyse/stable_runtime/canonical_master_20260523_audio_working_store_v1/canonical_calls_master.db" --ids-file product_data/customer_profiles/tz138_analyze_resolve_20260617/ids_100.txt --out-dir product_data/customer_profiles/tz138_analyze_resolve_20260617/analyze/gpt54_medium --cli /usr/bin/python3 --arms gpt54_medium:gpt-5.4:full --reasoning medium --prompt-profile full --sample-size 100 --provider codex_cli --timeout-sec 600 --keep-export-files
```

### gpt55_medium

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src PATH="/Applications/Codex.app/Contents/Resources:$PATH" LLM_CACHE_DIR="/Users/dmitrijfabarisov/Projects/Mango_tz138_analyze_sweep/product_data/customer_profiles/tz138_analyze_resolve_20260617/.llm_cache_gpt55_medium" python3 scripts/run_analyze_ab_test.py --source-db "/Users/dmitrijfabarisov/Projects/Mango analyse/stable_runtime/canonical_master_20260523_audio_working_store_v1/canonical_calls_master.db" --ids-file product_data/customer_profiles/tz138_analyze_resolve_20260617/ids_100.txt --out-dir product_data/customer_profiles/tz138_analyze_resolve_20260617/analyze/gpt55_medium --cli /usr/bin/python3 --arms gpt55_medium:gpt-5.5:full --reasoning medium --prompt-profile full --sample-size 100 --provider codex_cli --timeout-sec 600 --keep-export-files
```

### Resolve 25

```bash
DATABASE_URL="sqlite:////Users/dmitrijfabarisov/Projects/Mango_tz138_analyze_sweep/product_data/customer_profiles/tz138_analyze_resolve_20260617/resolve/resolve25.db" PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src PATH="/Applications/Codex.app/Contents/Resources:$PATH" LLM_CACHE_DIR="/Users/dmitrijfabarisov/Projects/Mango_tz138_analyze_sweep/product_data/customer_profiles/tz138_analyze_resolve_20260617/.llm_cache_resolve25" RESOLVE_LLM_PROVIDER=codex_cli CODEX_RESOLVE_MODEL=gpt-5.4-mini CODEX_REASONING_EFFORT=medium CODEX_CLI_TIMEOUT_SEC=600 python3 -m mango_mvp.cli resolve --limit 25
```

## Счётчики качества harness

`mini_high`:

- `summary_missing=0`
- `summary_looks_like_dialogue_dump=0`
- `summary_contains_english=0`
- `next_step_contains_english=0`
- `marked_non_conversation=1`
- `product_missing=25`
- `subjects_missing=19`
- `analysis_model_missing=0`
- `analysis_prompt_version_missing=0`

`gpt54_medium`:

- `summary_missing=0`
- `summary_looks_like_dialogue_dump=0`
- `summary_contains_english=0`
- `next_step_contains_english=1`
- `marked_non_conversation=1`
- `product_missing=28`
- `subjects_missing=18`
- `analysis_model_missing=0`
- `analysis_prompt_version_missing=0`

`gpt55_medium`:

- `summary_missing=0`
- `summary_looks_like_dialogue_dump=0`
- `summary_contains_english=0`
- `next_step_contains_english=1`
- `marked_non_conversation=1`
- `product_missing=31`
- `subjects_missing=19`
- `analysis_model_missing=0`
- `analysis_prompt_version_missing=0`

Эти счётчики не являются смысловым вердиктом. Смысловой регрейд по расшифровкам и JSON остаётся за Claude/Дмитрием.

## Стоимость и вызовы модели

- Analyze: 300 вызовов модели через Codex CLI subscription.
- Resolve: 0 вызовов модели на выбранном наборе, по отчёту CLI.
- Прямой API/OpenAI key не использовался.
- Прямая долларовая стоимость: 0 в рамках этого запуска, так как использовалась подписка Codex CLI.
- Точный расход токенов/денег Codex CLI текущий harness не отдаёт.

## Проверки безопасности

- Канонический путь БД: `/Users/dmitrijfabarisov/Projects/Mango analyse/stable_runtime/canonical_master_20260523_audio_working_store_v1/canonical_calls_master.db`.
- Контрольный stat после прогонов: `mtime_epoch=1781396422`, `size_bytes=1659514880`, `mtime_text=2026-06-14 03:20:22`.
- `git check-ignore` подтвердил, что `product_data/customer_profiles/tz138_analyze_resolve_20260617/...` игнорируется.
- Line count: 100 строк в каждом из трёх `analysis_json.jsonl`, 25 id в `resolve_ids_25.txt`, 25 `resolve.json` и 25 `transcript_text.txt`.
- Записей в AMO/Tallanto/CRM не было.
- ASR не запускался.
- Импорта в каноническую БД не было.

## Дальше

Следующий шаг — регрейд по сырью: сравнить три набора Analyze и 25 Resolve-выходов по расшифровкам, без доверия только формальным счётчикам.
