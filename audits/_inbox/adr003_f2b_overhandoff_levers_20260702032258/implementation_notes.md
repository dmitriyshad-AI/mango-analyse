# ADR-003 Ф2b Implementation Notes

Дата: 2026-07-02.

Ветка: `codex/adr003-semanticframe-migration`.

Ревизия анализа: `36ea110`.

## Что сделано

- Добавлен read-only отчётчик `scripts/report_adr003_overhandoff_levers.py`.
- Добавлены тесты `tests/test_report_adr003_overhandoff_levers.py`.
- Обновлён `docs/worktrees_registry.md`, потому что preflight остановился на двух фактических worktree, отсутствовавших в старом реестре.
- ТЗ перенесено в `tasks/_running/2026-07-02_TZ_ADR003_F2b_harmless_context_update_shadow_dlya_D1.md` и дополнено регрейдом Claude #1.
- Отчёт построен на M1-артефактах `36ea110`:
  `/Users/dmitrijfabarisov/Yandex.Disk.localized/OpenClaw/Actual Mango Tests/adr003_f2_clean_36ea110_20260702/runs/adr003_f2_self_answer_shadow_36ea110`.

## Ключевой результат

- Текущий route-handoff на gold safe/self: `11` строк (`manager_only=8`, `draft_for_manager=3`).
- Harmless ack/status route-only кандидаты: `0`.
- `frame_too_cautious`: `14`.
- В `frame_too_cautious` доминирует `check_availability`: `7`.
- Existence/format rows in frame-too-cautious: `8`.

Вывод: быстрый route-only Ф3 не найден. Реальный рычаг автономности лежит в различении "существует ли курс/формат" и "проверь живые места/запиши", но это требует fact-verification / anti-fabrication слоя.

## Почему это безопасно

- Скрипт только читает транскрипты и gold-report.
- Никакой runtime-код прямого пути не изменён.
- Никакие флаги, профиль, live, Telegram, AMO, CRM, Tallanto не трогались.
- `manager_only` используется только как диагностический источник over-handoff, не как active-кандидат.
