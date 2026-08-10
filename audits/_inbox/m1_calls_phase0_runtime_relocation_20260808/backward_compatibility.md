# Backward compatibility

- Runtime Calls не меняется до явного вызова нового relocation CLI.
- Смысл звонка, статусы, attempts, transcript/resolve/analyze и manager output
  сохраняются; меняются только известные локальные path-поля.
- SQLite schema и все business columns проверяются полным digest; отдельно
  проверяется полный вектор `(id, source_file)`.
- Ready DB и manifest остаются sealed и согласованными.
- Process A и Process B сохраняют прежние роли. M1 предназначен только для A,
  основной Mac — только для B.
- Несмотря на историческое имя функции, production stages уже выполняются
  последовательно: primary Whisper, GigaAM backfill, Resolve, Analyze.
- Старый source остаётся byte-identical; ручные checkpoint и manifest reseal
  запрещены.
- Новый `--check-sqlite` — отдельный read-only режим и не смешивается с
  inventory, dry-run или execute.
