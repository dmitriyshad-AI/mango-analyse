# Mango Calls: M4 handoff release

## Назначение

Эта ветка является единой точкой передачи актуального кода Mango Calls и проверенных ТЗ на MacBook M4. Она не является разрешением на production-deploy.

## Состав кода

История ветки линейна:

1. `ee6f20a3c47afd9a19b36c71199e06a161c252e3` — актуальная база Capture и безопасное масштабирование GigaAM workers.
2. `a65c6629` — GigaAM 0.2, прежняя v2 RNNT, batch=4 в отдельном окружении.
3. `0ef7ff99d5b740c49adc9fa542107c55a3e14821` — Pipeline `ProcessType=Interactive`.
4. `eb1c0321da75187e681588d90d56da3638f258ab` — детерминированный live Google publisher.

`eb1c0321` содержит все предыдущие пункты. Боевые worktree M1 при подготовке handoff не изменяются.

## Документы

- `README.md` — навигация и порядок пяти ТЗ.
- `TZ-01...TZ-05` — технические контракты.
- `EVIDENCE_MANIFEST.md` — воспроизводимые доказательства и ограничения.
- `MANGO_CALLS_PRODUCTION_AUDIT_AND_PLAN_2026-08-14.md` — расширенный аудит.
- `M4_CALL_DATABASE_HANDOFF_PROMPT.md` — безопасный обмен снимками баз.
- `M4_IMPLEMENTATION_PROMPT_2026-08-16.md` — задание Codex + Claude на M4.

## Конфиденциальные данные

В Git запрещены рабочие SQLite, аудио, publisher state, manager identity, credentials и env. Если для офлайн-проверки нужен конфиденциальный снимок, он передаётся отдельным DMG без пароля через личный каталог Яндекс.Диска владельца. Целостность проверяется по внешнему SHA-256 и внутреннему manifest.

Mango API credentials и Google service-account key в handoff не входят: M4 не получает права обращаться к живым системам.

## Граница полномочий

M4 работает в новой ветке/worktree и возвращает только код, тесты и аудиты. Любое изменение production M1 выполняется отдельно после независимого ревью и решения GO/STOP.
