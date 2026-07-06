> DONE 2026-07-06 21:25 | ветка codex/adr003-semanticframe-migration | codex

> TAKE 2026-07-06 20:41 | ветка codex/adr003-semanticframe-migration | codex

Ветка: codex/adr003-semanticframe-migration
Зоны: scripts/, src/mango_mvp/replay_exam/, tests/, tasks/, docs/, audits/_inbox/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_wappi_replay_machine_gate.py tests/test_wappi_replay_runner.py
Семантический-аудит: да

# Wrapper-ТЗ: deploy-readiness + replay-methodology v3

Источник: `/Users/dmitrijfabarisov/Claude Projects/Foton/2026-07-06_TZ_v3_deploy_i_replay_dlya_D1.md`.

Исполнить v3 с обязательными уточнениями аудита:

1. Поток A идет первым и не трогает live: read-only freeze из живой cwd, отдельные worktree `Mango_live_deploy@c262b954` и `Mango_live_rollback_eb6fa0b@eb6fa0b`, полный контракт отката, P0-smoke на `c262b954`, scrubbed-OUT и отчет в Foton.
2. Helper `scripts/adr003_deploy_swap_dry_run.py` не должен читать live heartbeat из candidate worktree. Либо добавить `--live-worktree`, либо freeze делать отдельными командами из live cwd, а helper использовать только для swap/rollback checklist.
3. Dynamic smoke запускать двумя отдельными прогонами, потому что `run_telegram_dynamic_client_sim.py` принимает один `--scenarios`. В обоих прогонах явно задавать `TELEGRAM_DIRECT_PATH_PILOT_CONFIG=pilot_gold_v1`, `--parallel 4`, `--judge-prompt-version v9.1`.
4. Поток R не двигает deploy-цель `c262b954`: экспорт pilot-10 в Foton только scrubbed; raw Wappi/PII остается локально; replay runner передает `client_safe_numbers` только из client-safe/retrieved фактов текущего хода. Запрещено брать числа из `manager_reference`, `raw_response`, `older_summary`, прошлых ходов или большого blob.
5. Добавить unit + integration tests для replay gate/runner.
6. До свапа сделать backup push ветки в origin без force.
7. M1 не запускать; live-write не выполнять; при двух неудачных итерациях остановиться и запросить решение.
