> DONE 2026-07-10 02:45 | ветка codex/calls-two-processes | codex

> TAKE 2026-07-09 22:26 | ветка codex/calls-two-processes | codex

Ветка: codex/calls-two-processes
Зоны: scripts/, src/mango_mvp/customer_timeline/, tests/, docs/, tasks/, audits/, .gitignore
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_mango_calls_two_processes.py tests/test_parallel_pipeline.py tests/test_customer_timeline_nightly_service.py
Семантический-аудит: нет

# Обработка звонков двумя процессами

Нормативный текст: `/Users/dmitrijfabarisov/Claude Projects/Foton/2026-07-09_CODEX_D4_TZ_calls_two_processes.md`, версия 3.0, включая обязательные правки аудита в разделе 7.

Дополнительные решения D4 после сверки исходного кода:

- базовая ревизия этой ветки: `00608cce`, потому что именно в ней находятся уже принятые staging-nightly, mango increment и snapshot-publish контуры; `origin/main` этих зависимостей пока не содержит;
- процесс A использует фактический `mango_office_capture_stage.py`, который сам делает read-only discovery и контролируемую загрузку, вместо неполной цепочки Stage 6 -> Stage 12, пропускающей asset/quarantine слои;
- процесс B выбирается один: готовый `mango_processed_summary` increment и импорт только в timeline-staging; prod timeline не меняется;
- без лимита означает дренаж очереди до нескольких пустых циклов, а не бесконечный размер одного батча;
- реальные ASR и Resolve+Analyze разрешены владельцем только внутри процесса A, без sync/CRM/Tallanto;
- приёмочные и ежедневные отчёты содержат только агрегаты и маскированные примеры.
