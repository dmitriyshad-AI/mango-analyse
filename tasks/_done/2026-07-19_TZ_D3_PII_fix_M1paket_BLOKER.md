> DONE 2026-07-19 22:10 | ветка main | codex

> TAKE 2026-07-19 21:39 | ветка main | codex

Ветка: main
Зоны: src/mango_mvp/replay_exam/pseudonymizer.py, tests/, tasks/, audits/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_wappi_replay_pseudonymizer.py tests/test_wappi_replay_pii_scan.py tests/test_wappi_replay_provider_adapter.py
Семантический-аудит: да

# D3: закрыть PII-блокер M1-пакета

Каноническое ТЗ:
`/Users/dmitrijfabarisov/Claude Projects/Foton/2026-07-19_TZ_D3_PII_fix_M1paket_BLOKER.md`.

Исправить детектор и маскирование международных телефонов, добавить
регрессионные тесты, пересобрать только очищенный Wappi-набор во внешнем
M1-пакете и пересчитать его манифесты. M1 не запускать до независимого
нулевого перескана. Бот, live и внешние системы не менять.
