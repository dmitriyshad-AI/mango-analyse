# Промпт для Codex на MacBook M4

Скопируйте весь текст ниже в основную задачу Codex на M4. Он специально разделён на безопасный read-only инвентарь и создание пакета только после отдельного подтверждения.

```text
Ты работаешь на MacBook M4 с production-данными звонков. Цель — подготовить доказуемую передачу всех слоёв call database на MacBook M1, не повредив и не остановив production.

Задача состоит из двух строго разделённых этапов. Сначала выполни только read-only аудит и покажи владельцу обезличенный агрегатный отчёт. На этапе 1 разрешена только запись этого одного sanitized Markdown-отчёта; запрещены row-level crosswalk, file-level manifest, backup, архив и передача данных. Не переходи к ним, пока Дмитрий явно не подтвердит этап 2.

Перед началом:

1. Найди и полностью прочитай все применимые AGENTS.md от корня проекта до рабочей папки. Следуй более близкому файлу при конфликте.
2. Определи активный worktree, commit SHA и фактическую production-конфигурацию.
3. Не останавливай, не перезапускай и не перенастраивай службы.
4. Не изменяй код, базы, очереди, статусы, конфигурацию, Google и launchd.
5. Не запускай второй экземпляр обработчика, capture, publisher или другого writer.
6. Не показывай в чате телефоны, ФИО, тексты/цитаты разговоров, аудио, filenames с ПДн, токены, пароли и ключи.
7. Для единственного отчёта этапа 1 выбери локальный owner-only каталог вне Git и вне iCloud/Yandex.Disk/Dropbox/OneDrive/Google Drive/общих папок; проверь `realpath`, отсутствие symlink, текущего owner UID и режим каталога `0700`, файла `0600`. При сомнении остановись до записи.

## Этап 1. Только read-only инвентарь

Найди все слои хранения звонков, включая, но не ограничиваясь:

- исторический `canonical_calls_master.db`;
- рабочую `mango_calls_pipeline.sqlite` / mutable queue;
- sealed `mango_calls_ready.sqlite` / drop и manifest;
- Customer Timeline SQLite, куда импортировались события звонков;
- другие SQLite/JSONL/manifest/индексы и каталоги audio, относящиеся к звонкам.

Известные прежние числа (примерно 65 974 canonical calls, 5 248 working, 5 096 sealed) используй только как гипотезу. Проверь независимо; не подгоняй результат.

Для каждого слоя собери без ПДн:

- логическое назначение и какой процесс пишет/читает;
- абсолютный путь в локальном owner-only отчёте; в чате маскируй username;
- размер, mtime, наличие `-wal/-shm/-journal`;
- `PRAGMA user_version`, schema tables/views/indexes/triggers;
- безопасный `PRAGMA quick_check` в read-only режиме;
- число звонков/событий и распределение статусов Capture, ASR, GigaAM, Resolve, Analyse, Google/errors;
- минимальную/максимальную дату звонка и максимальный `updated_at`;
- уникальность и дубли доступных устойчивых ключей;
- tenant, provider, provider_call_id/source_call_id, canonical_call_id, local id;
- версии ASR/Resolve/Analyse/prompt/normalizer/projection, если сохранены;
- branch, commit SHA, clean/dirty worktree, Python/model/library versions;
- launchd labels/ProgramArguments/WorkingDirectory без секретов.

На большой live SQLite не запускай тяжёлый полный `integrity_check`, если он может мешать production. На этапе 1 достаточно безопасного quick_check и отчёта о размере/риске. Полный integrity_check обязателен на консистентной backup-копии этапа 2.

## Пересечения и пробелы

Посчитай сравнение баз в памяти/read-only SQL по ключу:

`tenant + provider + provider_call_id`

Если его нет, явно опиши резервный ключ и риск. Не объединяй по local id, filename, телефону+времени или canonical autoincrement без crosswalk.

Покажи только агрегаты и digest наборов ключей, без самих row-level идентификаторов:

- только M4 canonical;
- только M4 working/drop;
- присутствует в нескольких слоях;
- совпадает по provider call ID, но различается статус/текст/version/audio SHA;
- без provider call ID;
- конфликт или дубль.

Проверь как гипотезы:

- возможное пересечение M4↔текущего M1 около 260 звонков;
- возможный пробел истории между 22 мая и 8 июля (год установи по данным).

Если данных M1 на M4 нет, не выдумывай сравнение: на этапе 1 опиши только схему будущих manifest/crosswalk, aggregate key coverage и digest, чтобы M1 Codex понял план после передачи.

Будущий raw crosswalk этапа 2 должен включать tenant/provider/provider_call_id/canonical_call_id/source layer/status/audio hash или наличие конфликта. На этапе 1 его не создавай и row-level идентификаторы никуда не записывай; реальные идентификаторы никогда не вставляй в чат.

## Конфигурация и аудио

Собери sanitized config: названия моделей, версии prompt/schema, timezone, tenant/provider, пути и hashes справочников. Значения секретов заменяй `<REDACTED>`; не копируй service-account JSON/cookies/tokens.

На этапе 1 аудио не копируй. Дай только агрегаты: count, общий размер, расширения, date range, linked/unlinked/missing. File-level audio manifest и сами записи — только после отдельного разрешения этапа 2.

## Итог этапа 1

Сохрани локальный owner-only `M4_CALL_DB_INVENTORY_REPORT.md` без ПДн и выведи кратко:

1. Какие слои найдены.
2. Что является canonical source, mutable queue, sealed handoff и Timeline.
3. Counts/date ranges/statuses/integrity.
4. Overlap/duplicates/conflicts/gaps.
5. Code/model/schema versions.
6. Что именно рекомендуешь передать на M1 и оценочный размер.
7. Что исключено: secrets/audio/ПДн.

После этого остановись и задай ровно вопрос:

«Read-only инвентарь завершён. Разрешаете этап 2: создать консистентные DB snapshots, raw manifests/crosswalk и sanitized config в локальном пакете без пароля для передачи через ваш Яндекс.Диск? Аудио и file-level audio manifest по умолчанию не включаю и запрошу отдельно».

Не переходи к этапу 2 без явного ответа Дмитрия.

## Этап 2. Только после явного подтверждения

Создавай SQLite snapshots только SQLite Backup API (или эквивалентным онлайн-backup способом с доказанной консистентностью).

Разрешение этапа 2 должно отдельно перечислять категории: DB snapshots, raw DB/file manifests, raw crosswalk и sanitized config. Аудио и file-level audio manifest требуют третьего отдельного подтверждения; общее «передай всё» не считается разрешением на аудио.

До создания файлов:

1. Оцени суммарный размер source DB и будущих snapshot/crosswalk/temp/DMG.
2. Проверь свободное место на локальном M4-томе; требуй не меньше `2 × ожидаемый размер пакета + 20%`, иначе STOP.
3. Создай уникальный локальный owner-only staging вне Git, облачных/синхронизируемых/общих папок и сетевых томов. Проверь `realpath`, отсутствие symlink/cloud-marker, owner UID, каталог `0700`, файлы `0600`; любое несоответствие — STOP.
4. Собери обычный read-only DMG без шифрования и пароля. Передавай его только через явно утверждённый личный каталог Яндекс.Диска Дмитрия. Рядом положи отдельный файл `.sha256`; внутри пакета должен быть полный manifest с SHA каждого файла.

Запрещено:

- `cp`, Finder, архиватор или `rsync` живой `.sqlite`;
- копирование живых `-wal/-shm`;
- архив непосредственно из каталога working DB;
- остановка служб без отдельного разрешения;
- второй writer;
- любые UPDATE/INSERT/DELETE/VACUUM/migration;
- передача пакета через другое облако, публичную ссылку, Git или общий каталог;
- передача DMG не через явно утверждённый личный каталог Яндекс.Диска Дмитрия.

Для каждой backup-копии в проверенной staging-папке:

1. Сделай online SQLite backup.
2. На копии выполни quick_check и полный integrity_check.
3. Зафиксируй SHA-256, размер и UTC/MSK snapshot time.
4. Сохрани schema dump без данных.
5. Пересчитай counts/status/date ranges/unique keys.
6. Создай crosswalk и conflict report.
7. Добавь sanitized config, commit и версии.
8. Убедись, что WAL/SHM и секреты не попали.

Структура:

- `README.md` — назначение и восстановление;
- `MANIFEST.json` — все files, sizes, SHA-256, provenance;
- `DATABASES/` — консистентные SQLite backups;
- `SCHEMA/` — schema без данных;
- `REPORTS/` — агрегаты/overlap/gaps/conflicts;
- `CROSSWALK/` — устойчивые идентификаторы;
- `CONFIG_SANITIZED/` — без секретов;
- `CODE_VERSION/` — commit/dependencies/models;
- `AUDIO_MANIFEST/` — только если отдельно разрешено.

Для каждой базы MANIFEST содержит logical role, masked source path, backup filename, size, SHA, snapshot UTC/MSK, user_version, quick/integrity result, record count/date range/status counts/key set/code versions.

После создания собери MANIFEST и per-file SHA, затем создай read-only DMG без пароля и отдельно зафиксируй его SHA/размер. Покажи Дмитрию masked local path, состав, размер, SHA и проверки. После явного разрешения Дмитрия перемещай DMG и файл `.sha256` в утверждённый личный каталог Яндекс.Диска.

До приёма M1 отдельно проверяет свободное место по той же формуле. DMG из Яндекс.Диска копируется в локальный owner-only staging и только там открывается read-only. Пароль не требуется. Порядок receipt строгий:

1. сверить SHA-256 и размер DMG до открытия;
2. открыть DMG read-only, не изменяя исходный объект;
3. сверить неизменённый MANIFEST, SHA-256 и размер каждого файла;
4. выполнить quick_check/integrity_check каждой SQLite-копии;
5. сверить counts/date ranges/key digests;
6. сохранить owner-only `M1_M4_PACKAGE_RECEIPT.md` с PASS/STOP без ПДн.

Любое несовпадение — STOP без merge. Слияние в production/Timeline не выполнять автоматически: сначала union preview, overlap/gap/conflict report и отдельное утверждение плана. M1 остаётся единственным live-writer; M4 snapshots неизменяемы/read-only, обратный merge M1→M4 и два live writer запрещены.

Lifecycle: production sources на M4 не удаляются никогда этим заданием. Локальный staging и snapshots не удалять до письменного M1 acceptance. После acceptance Дмитрий отдельно утверждает очистку временного staging; фиксируется delete receipt с перечнем хэшей, а DMG в Яндекс.Диске хранится до отдельного решения Дмитрия. При любом STOP доказательства сохраняются owner-only, автоматическая очистка запрещена.
```

Официальная справка по применению `AGENTS.md`: [OpenAI Codex — Custom instructions with AGENTS.md](https://developers.openai.com/codex/guides/agents-md).
