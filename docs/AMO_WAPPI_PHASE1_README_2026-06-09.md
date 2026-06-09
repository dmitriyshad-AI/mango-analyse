# AMO + Wappi Phase 1

1. Секреты хранить только вне репозитория: `~/.mango_secrets/amo_wappi.env`, права `600`; минимально нужны `AMOCRM_BASE_URL`, `AMOCRM_ACCESS_TOKEN`, `WAPPI_TELEGRAM_TOKEN` и/или `WAPPI_MAX_TOKEN`.
2. Карту `profile_id -> brand` и allowlist одной тестовой сделки держать в `~/.mango_secrets/amo_wappi_phase1.json`; пример без секретов: `config/amo_wappi_phase1.example.json`.
3. Проверка без live-write: `PYTHONPATH=src python3 - <<'PY'\nfrom mango_mvp.integrations.amo_wappi_phase1 import load_env_file, AmoWappiPhase1Config\nload_env_file()\nprint(AmoWappiPhase1Config.from_file())\nPY`

Фаза 1 не содержит автоотправки клиенту. Единственная write-функция пишет внутреннее примечание только в сделку из `allowed_test_lead_ids`; остальные `lead_id` блокируются до HTTP-запроса.
