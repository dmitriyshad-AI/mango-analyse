# TZ-155 Report — light git bundles and job manifests

Дата: 2026-06-30  
Ветка: `codex/tz155-light-git-bundles`  
Коммит реализации: `d739cc9c9181bef5325dba5e0eb1b8198930932a`

## Что сделано

- Создан `scripts/build_job_manifest.py`: пишет маленький `job_<sha>.json` с `commit_sha`, `set_rel_path`, `set_sha256`, `snapshot_rel_path`, `snapshot_sha256`, `env_flags`, `parallel`, `max_hours`, `run_cmd`.
- Добавлен runbook ручного M1-пути: `docs/M1_GIT_JOB_MANIFEST_RUNBOOK_2026-06-30.md`.
- `scripts/build_mango_clean_bundle.py` оставлен как explicit fallback: CLI теперь требует `--allow-heavy-bundle`.
- `scripts/m1_watcher.py` помечен как dormant/deprecated для новых быстрых eval.
- В git добавлен eval-набор `product_data/telegram_dynamic_test_sets/forward_payment_personas_20260630.jsonl`.
- Уже в локальном `main` были отслеживаемые наборы:
  - `product_data/telegram_dynamic_test_sets/p0_stability_set_20260617.jsonl`;
  - `product_data/telegram_dynamic_test_sets/adr003_semantic_frame_wappi_latest25_20260630.jsonl`.

Примечание: `p0_stability_set_20260617.jsonl` содержит только `persona` rows без `simulator_spec/judge_spec`, поэтому как одиночный `--scenarios` для `run_telegram_dynamic_client_sim.py` невалиден. Новый forward-payment набор валиден.

## Git/Yandex

- Создан bare repo: `~/Yandex.Disk.localized/OpenClaw/mango_repo.git`.
- Добавлен remote `yandex` на главном маке.
- Выполнено `git push yandex --all`.
- Размер зеркала: `30M`; `git count-objects -vH`: `size-pack: 29.59 MiB`.
- Размер smoke job manifest: `4.0K`.

## Проверка нового пути

Манифест:

`~/Yandex.Disk.localized/OpenClaw/Actual Mango Tests/_jobs/job_d739cc9c9181.json`

Fresh clone:

`/tmp/mango_tz155_clone.6YrgRK`

Проверки:

- `HEAD == d739cc9c9181bef5325dba5e0eb1b8198930932a`;
- set sha256 совпал: `5c8907fc1730c96240bf9618be1f1774bfe29e25cc4a0c30d13e95f5d90b2187`;
- snapshot sha256 совпал: `3ee26efcee8f6af2060767a637ed3aa6a82faae94a335768bd6568e064d39aa5`;
- smoke run: `--limit 1`, `parallel=1`, out-dir `runs/20260630_tz155_smoke_limit1`;
- результат smoke: `dialogs=1`, `turns=2`, `PASS=1`, `hard_gate_failures=0`, `config_validity.invalid=false`;
- `llm_calls.total=9`.

## Тесты

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_build_job_manifest.py tests/test_m1_watcher.py
```

Результат: `28 passed in 1.00s`.

Дополнительно:

```bash
python3 -m py_compile scripts/build_job_manifest.py scripts/build_mango_clean_bundle.py scripts/m1_watcher.py
```

Результат: OK.

## Секреты

Проверено:

- tracked sensitive path check по `auth.json`, `.codex/`, `.env`, `.env.private`: пусто;
- mirror sensitive path check по тем же путям: пусто;
- token-pattern grep по рабочей ветке и mirror нашёл только плейсхолдеры/тестовые фейки:
  - `OPENAI_API_KEY=...`;
  - `amo-token`;
  - `wappi-token`;
  - fake Telegram token в тесте маскирования.

Реальные `auth.json`, `~/.codex`, API tokens в git/mirror не обнаружены.

## Границы

- Живой бот, CRM, AMO, Tallanto не трогались.
- Текущий тяжёлый M1 A/B/C прогон 153/154 не трогался.
- Вотчер M1 не запускался и не оживлялся.
