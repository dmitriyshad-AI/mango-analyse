# D1 Project Registry Docs

Дата: 2026-06-26

Ветка: `codex/project-registry-docs-current`
Base: `main@b543991`

## Что сделано

Добавлен repo-local реестр:

- `docs/PROJECT_REGISTRY.md`

Реестр фиксирует текущие слои проекта:

- local `main`;
- `origin/main`;
- live runtime;
- branch candidates;
- gates перед Wappi, AMO card writeback и Customer Timeline apply.

## Проверки

Self-review:

- `main` указан как `b543991`;
- `origin/main` указан как `43134ae`, local main ahead by 89;
- Wappi branch указан как `codex/wappi-watch-on-main` @ `29c1dee`;
- P0 branch указан как `codex/p0-three-classes-on-main` @ `9558b1f`;
- live Wappi loop не назван живым: указан stale heartbeat `2026-06-25T17:46:09Z`, `bot_calls=0`, `processed=0`;
- registry не содержит утверждения `production-ready`;
- registry явно не авторизует live-write.

## Semantic Review

Verdict: `PASS_WITH_NOTES`.

Реестр годится как текущая карта проекта, потому что разделяет `main/live/branch` и не превращает formal-pass в разрешение на production. Остаточный риск: файл нужно обновлять после каждого accepted merge, runtime deployment или live-write решения.

## Safety

- Код не менялся.
- AMO/Tallanto/CRM write: 0.
- Live bot / `stable_runtime` не трогались.
