# Mango Calls Phase 0 runtime relocation

Вердикт: **GO только для синтетической кодовой Фазы 0**. Host-readiness,
реальный перенос и live-cutover остаются **STOP**.

Пакет привязан к implementation tree
`86cf64a33e4034234e334ae742e57d51b9a7cf72` и staged diff SHA-256
`1d0ade925d238a201706fe64d21b86e74695e48d257266c5ef87251eed99718f`.
Родитель ветки — `82208ad1e2c95ca0c8476ec3e9b88268ebb3d455`, исходный Calls handoff —
`f8faabf1d442261023605ee3285deb3b2a278cf9`.

Реальные Mango API, ASR, Resolve, Analyze, launchd, cutover, stable_runtime,
CRM/Tallanto/Wappi и клиентские сообщения не запускались. Source и пакет
звонков не изменялись.

Финальная точная проверка Claude CLI не состоялась из-за месячного spend
limit. Это ограничение зафиксировано явно; exact tree независимо проверили
Codex architect, breaker, business-auditor и cleaner.
