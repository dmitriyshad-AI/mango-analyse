# Tests

## Авторитетный Calls-профиль

Команда охватила capture/schedule/handoff/two-processes/publication/readiness,
relocation и stage verdict тесты.

Результат: `746 passed, 42 warnings in 45.24s`.

## Controlled/regression-профиль

Команда охватила controlled scope, bootstrap, claims, four-stage services,
dialogue и ASR drop-защиту.

Результат: `413 passed, 425 warnings, 10 subtests passed in 24.32s`.

Отдельный файл controlled-one независимо воспроизведён Codex cleaner и Claude:
`48 passed`.

## Отрицательные контроли

- controlled proof не пишет shared marker;
- service pipeline и Process A `--skip-capture` после него остаются STOP;
- stale proof, cursor mismatch и manifest race fail-closed;
- ticket содержит digest проверенных bytes, worker перечитывает manifest;
- allowlist rejects 0/2 IDs, digest drift, symlink, hardlink и чужой host;
- claims/recovery не затрагивают non-target;
- malformed/missing/boolean/negative stage metrics не дают PASS;
- snapshot missing/tamper, unlink error и непустой run-dir сохраняют evidence,
  дают failed/pilot=false, а остаток блокирует повтор;
- cleanup `OSError` не маскирует исходную stage `RuntimeError`;
- обычный успех и идемпотентный нулевой повтор не регрессировали.

`py_compile`, Python 3.12 AST и `git diff --check` прошли. Полный тестовый
профиль выполнялся локальным Python 3.14; для Python 3.12 на машине не был
подготовлен полный набор зависимостей, поэтому runtime-pass не заявляется.
