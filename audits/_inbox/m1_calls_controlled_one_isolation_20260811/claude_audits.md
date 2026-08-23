# Claude CLI audits

CLI проверен из корня репозитория:

- Claude Code `2.1.223`;
- `--help` подтвердил aliases `opus`, `fable`, `sonnet` и effort `xhigh`;
- `claude auth status`: `loggedIn=true`;
- все сессии read-only, без Edit/Write/Web и без реальных Calls-операций.

## Opus 5 — архитектура

Нашёл два практических дефекта cleanup evidence: потерю полного отчёта при
неудачном `rmdir` и маскирование primary stage exception через cleanup
`OSError`. Оба исправлены и повторно проверены. Финал: оба дефекта закрыты,
P0/P1 к коммиту нет; pilot и service остаются STOP до разрешения/доказательств.

## Sonnet 5 — breaker

Финальные 8 SHA совпали. Воспроизведено `301 passed`; дополнительно выполнены
атаки на proof/ticket/manifest и cleanup. Service не разблокирован, cleanup не
создаёт PASS и не маскирует primary error. Финал: GO, P0=0.

Sonnet отметил replay ticket внутри узкого окна тем же UID. Это принято как P2:
ticket защищает от случайного/устаревшего запуска внутри документированной
same-UID trust boundary, а не от произвольного процесса владельца аккаунта.
Nonce не создаст криптографической изоляции от того же UID.

## Fable 5 — бизнес

Финальные SHA совпали, controlled file `48/48`. Вердикт: GO к коммиту, но
честный бизнес-статус STOP — обработанных новым сервисом реальных звонков и
доставленной менеджерам пользы пока нет. Следующий путь: чистый SHA -> read-only
probe -> allowlist -> отдельное разрешение на один звонок -> ручная сверка ->
нулевой повтор.
