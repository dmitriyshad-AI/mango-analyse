# Implementation notes

## Корневое решение

Relocation работает только с явными `old_root`, `new_root`, `pipeline_root` и
полным source inventory. Локальный runtime строится от реального `$HOME` и
разрешён только внутри owner-only `$HOME/.mango_local`, вне Git и cloud-sync.

Скрипт не выполняет слепую замену строк в SQLite. Он знает только runtime
callers и меняет:

- разрешённые path-поля capture JSONL;
- `call_records.source_file` в working и ready DB;
- `ready_db` и storage seal в ready manifest.

Все другие поля и таблицы являются отрицательным контролем.

## Надёжность

- inventory до/после содержит hashes, sizes, modes и timestamps;
- межмашинная source/target-сверка сопоставляет пути, размеры, SHA-256, modes и
  totals; `mtime_ns` намеренно не входит в transfer-проекцию;
- durable plan, staging и complete лежат в owner-only state вне pipeline;
- повтор завершённой операции возвращает `already_relocated` и `changes=0`;
- незавершённый commit продолжается только из доказанного before/after
  поколения;
- оборванная последняя JSONL-запись сохраняется и восстанавливается тем же
  runtime-механизмом;
- symlink, special file, внешний hardlink, чужой владелец или небезопасная смена
  inode дают fail-closed STOP;
- обычные отклонения runtime modes видны в dry-run как `permissions_to_change`;
  execute нормализует каталоги/файлы до `0700/0600` только под проверками
  identity, links, size, `mtime_ns` и SHA-256.

Source не checkpoint-ится и не reseal-ится вручную. Непустой WAL означает STOP
и отдельное разрешение владельца на штатный Process A republish без capture,
ASR, Resolve и Analyze, после чего все проверки начинаются заново.

`--check-sqlite` использует encoded file URI, `mode=ro&immutable=1`, memory-only
temp store и exact quick/integrity checks. До открытия он отклоняет active WAL,
rollback journal, unsafe sidecar и неверный SHM; до/после сравниваются inode,
link count, owner, size, mtime, mode и SHA-256.
Финальный порядок — DB SHA-256, затем sidecar-snapshot, затем main-DB `lstat`;
так late WAL не теряется во время долгого hash-чтения.

## Graphify

Graphify использован только для навигации. Exact parent map для
`82208ad1e2c95ca0c8476ec3e9b88268ebb3d455` имел SHA-256
`61a5aa7e2dd805106ce3f9187a1d15c1f6d88aa6e2fea6bbdb016c8ee3675acb`,
221 copied files, 29 structured files, 7011 nodes и 27986 edges. После коммита,
меняющего `scripts/`, карта должна быть пересобрана на exact commit SHA вне Git
до push.
