> DONE 2026-07-16 13:58 | ветка codex/integrate-d3-d4-20260712 | codex

> TAKE 2026-07-16 13:42 | ветка codex/integrate-d3-d4-20260712 | codex

Ветка: codex/integrate-d3-d4-20260712
Зоны: deploy/customer_timeline_daily_captures/, deploy/customer_timeline_nightly/, docs/worktrees_registry.md, scripts/, src/mango_mvp/channels/pilot_profile_runtime.py, src/mango_mvp/customer_timeline/, src/mango_mvp/integrations/, src/mango_mvp/services/, tests/, tasks/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_amo_wappi_auto_resolver.py tests/test_run_amo_wappi_draft_loop.py tests/test_wappi_history_import_to_timeline.py tests/test_draft_loop.py tests/test_customer_timeline_codex_task.py tests/test_customer_timeline_nightly_service.py tests/test_mango_calls_two_processes.py tests/test_ingest_filename_parse.py tests/test_telegram_dynamic_client_sim.py
Семантический-аудит: да

# Окно 2: исправленный точечный перенос

Источник решения:
`/Users/dmitrijfabarisov/Claude Projects/Foton/2026-07-16_TZ_Okno2_cherry_pick_i_poryadok.md`.

## Объём

1. Перенести целиком строгую бренд-защиту из `d80770a6`.
2. Из `d2b74d5d` перенести только сохранение настроенного транспорта AMO и тест.
3. Из `35299f23` перенести runtime identity и строгий профиль симулятора, но исключить W3, `post_layers.py`, W3-тесты и новый флаг.
4. Из `2639f55f` перенести финальную версию Wappi hints и устойчивость служб, звонков и ingest; сохранить бренд-защиту шага 1.
5. Из `ebcbe23d` перенести nightly convergence; в nightly-конфликтах использовать новую версию, а в Wappi-brand оставить общий guard шага 1.

## Не входит

- `TELEGRAM_TONE_CLOSE_GATE_FINDINGS_FLOOR` и любой W3-код;
- переключение или остановка live-служб;
- AMO, Tallanto, CRM или Wappi write;
- удаление файлов, баз, веток или worktree;
- перенос отчётов и устаревшего `docs/worktrees_registry.md` из веток-источников.

## Приёмка

- В коде отсутствует `TELEGRAM_TONE_CLOSE_GATE_FINDINGS_FLOOR`.
- Wappi resolver fail-closed на отсутствующем, неизвестном, смешанном и чужом бренде.
- Read-only Wappi transport не расширяет разрешённые хосты и сохраняет настроенный transport.
- Автоматическая Wappi-пара не получает bot-safe память; resolver остаётся выключен.
- Nightly использует постоянный runtime-root и проверяемую базу обработанных звонков.
- Целевые тесты и полный безопасный pytest зелёные.
- P0, бренд, ПДн и анти-выдумка не ослаблены.

## СТОП

- Основной worktree перестал быть чистым из-за чужих изменений.
- Нужен live-write, перезапуск службы или доступ к секретам.
- После двух попыток не удаётся согласовать конфликт без ослабления защитного пола.
- Целевой или полный тест показывает регрессию P0, бренда, ПДн или фактов.
