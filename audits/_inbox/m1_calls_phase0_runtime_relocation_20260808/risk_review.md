# Risk review

## Закрытые риски

- hardcoded source username в Calls docs/env заменён на `$HOME` или явные
  параметры source/M1/main;
- missing DB больше не создаётся проверкой;
- raw SQLite output не считается успехом;
- writable checkpoint source удалён из runbook;
- invalid ready seal, active WAL, rollback journal, unsafe sidecar и bad SHM
  дают STOP;
- symlinked parent и внешний WAL hardlink отклоняются полным inventory до
  передачи;
- URI со space, `?`, `#` и percent-последовательностями кодируется безопасно;
- все три legacy launchd labels должны быть остановлены, оба process locks —
  свободны;
- full inventory и recovery artifacts запрещены в Git/audit pack.

## Остаточные ограничения

- Фактический `$HOME/.mango_local` на M1 сейчас имеет mode `0755`, тогда как
  контракт требует `0700`; никакой chmod не выполнялся.
- Runtime venv отсутствует; `mlx-whisper`, `gigaam` и `imageio-ffmpeg` не
  установлены. Установка не выполнялась.
- Состояние live WAL, launchd и source Mac не проверялось и остаётся STOP.
- Handoff package является snapshot, а не authoritative ответом текущего Mango
  API; по нему нельзя утверждать, что серверных дублей нет.
- Финальный exact Claude CLI audit заблокирован месячным spend limit. Ранее
  Claude Sonnet нашёл fail-open SQLite checks; вывод был воспроизведён и закрыт.
  Exact финальное дерево независимо проверено четырьмя Codex-ролями.
- Код читает некоторые небольшие SQLite/JSON artifacts целиком; для текущего
  handoff это допустимо, но остаётся неблокирующим ограничением масштаба.

Активно вредоносный процесс того же UID вне threat model. Любой фактический
host drift, non-zero WAL, inventory mismatch или seal mismatch означает STOP.
