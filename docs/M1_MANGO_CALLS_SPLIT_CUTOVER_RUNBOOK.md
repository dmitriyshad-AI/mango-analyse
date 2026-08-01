# Перенос обработки звонков на M1

## Целевая схема

- M1 выполняет только Process A: Mango API, загрузка, Whisper, GigaAM, Resolve,
  Analyze, проверенная ready-база и суточный отчёт.
- Основной Mac сам забирает ready-базу с M1 по защищённому SSH и выполняет только
  Process B рядом с актуальной базой истории клиентов.
- M1 не получает права записи или запуска команд на основном Mac.
- SQLite, аудио, рабочие базы и секреты не передаются через Git или Яндекс Диск.
  Git передаёт только код. По прямому требованию владельца Яндекс Диск получает
  итоговые внутренние XLSX и TXT; они содержат персональные данные, поэтому
  доступны только владельцу и допущенным сотрудникам.

## Что блокирует переключение сейчас

Нельзя начинать перенос, пока на основном Mac активен Process A или есть живые
заявки на обработку. Сначала текущий цикл должен штатно закончиться. Остановка
или перенос работающей SQLite запрещены.

## Проверка обоих компьютеров

1. На обоих компьютерах одна и та же чистая ревизия Git:

   ```bash
   git fetch origin
   git switch main
   git pull --ff-only origin main
   test -z "$(git status --porcelain)"
   git rev-parse HEAD
   ```

   Полученные SHA должны совпасть с SHA, указанным владельцем при переключении.

2. На M1 свободно не меньше 40 ГиБ, установлен `ffmpeg`, работает Python-среда
   с `requirements.txt`, `requirements-local-whisper.txt` и
   `requirements-local-dual-asr.txt`, локально выполнен вход в Codex.
3. Секреты создаются вручную только в `~/.mango_secrets/`, каталоги имеют режим
   `0700`, файлы `0600`. В Git, Яндекс Диск и отчёты секреты не попадают.
4. На M1 установлен клиент Яндекс Диска. Google-ключ хранится отдельно с режимом
   `0600`; папка Google Drive непубличная и дана только нужным сотрудникам и
   служебной учётной записи.
5. Основной Mac фиксирует ключ хоста M1 в отдельном `known_hosts`. M1 не имеет
   SSH-ключа к основному Mac. Для чтения ready-drop используется отдельный ключ
   текущего пользователя M1 с принудительной read-only командой из раздела ниже.

## Первичная передача рабочего состояния

Передача выполняется один раз с основного Mac напрямую на M1 по SSH, только
после штатного завершения Process A. Сохраняется тот же абсолютный путь, потому
что сохранённые записи содержат абсолютные пути к аудио:

Сначала сохранить доказательства для отката:

```bash
SNAP="$HOME/.mango_local/mango_calls_cutover/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$SNAP" && chmod 700 "$SNAP"
printf '%s\n' "$SNAP" > "$HOME/.mango_local/mango_calls_cutover/MAIN_SOURCE_SNAPSHOT_PATH"
chmod 600 "$HOME/.mango_local/mango_calls_cutover/MAIN_SOURCE_SNAPSHOT_PATH"
launchctl print "gui/$(id -u)/com.mango.calls-process-a" > "$SNAP/process-a.before.txt" 2>&1 || true
launchctl print "gui/$(id -u)/com.mango.calls-process-b" > "$SNAP/process-b.before.txt" 2>&1 || true
launchctl print "gui/$(id -u)/com.mango.calls-two-processes" > "$SNAP/legacy.before.txt" 2>&1 || true
cp -p ~/Library/LaunchAgents/com.mango.calls-process-*.plist "$SNAP/" 2>/dev/null || true
cp -p ~/.mango_local/mango_calls_two_processes/config.json "$SNAP/"
shasum -a 256 \
  "/Users/dmitrijfabarisov/Projects/Mango analyse/product_data/mango_calls_two_processes/drop/"* \
  > "$SNAP/drop.sha256"
```

Затем передать данные. Встроенный macOS rsync 2.6.9 не поддерживает
`--info=progress2`, поэтому используется совместимый `--progress`; пробел в
удалённом пути экранирован для удалённой оболочки:

```bash
M1_HOST='dmitrijfabarisov@ИМЯ-ИЛИ-IP-M1'
/usr/bin/rsync -aH --progress \
  "/Users/dmitrijfabarisov/Projects/Mango analyse/product_data/mango_calls_two_processes/" \
  "$M1_HOST:/Users/dmitrijfabarisov/Projects/Mango\\ analyse/product_data/mango_calls_two_processes/"
```

После первой копии Process A на основном Mac остаётся выключенным, выполняется
короткая повторная `rsync -aH`, затем на M1 проверяются:

```bash
PIPELINE='/Users/dmitrijfabarisov/Projects/Mango analyse/product_data/mango_calls_two_processes'
chmod 700 "$PIPELINE" "$PIPELINE"/{capture,working,drop,state,locks,reports,logs}
find "$PIPELINE" -type f \( -name '*.sqlite' -o -name '*.sqlite3' -o -name '*.json' \) -exec chmod 600 {} +
sqlite3 "$PIPELINE/working/mango_calls_pipeline.sqlite" 'PRAGMA quick_check; PRAGMA integrity_check;'
sqlite3 "$PIPELINE/drop/mango_calls_ready.sqlite" 'PRAGMA quick_check; PRAGMA integrity_check;'
```

`stat -f '%Su:%Lp %N'` должен показать текущего пользователя и `700` для
перечисленных каталогов, `600` для SQLite/JSON. Оба результата SQLite должны
быть `ok`. База истории клиентов на M1 не копируется:
Process B остаётся на основном Mac.

### Отдельный SSH-ключ только на чтение

Чтобы не создавать и не настраивать системного пользователя вслепую, основной
Mac использует отдельный SSH-ключ. На M1 этот ключ разрешён только с
принудительной read-only командой: произвольная команда и доступ к секретам
через него невозможны. В `/Users/Shared` создаются root-owned ссылка и копия
проверяющего скрипта:

```bash
sudo cp scripts/mango_calls_readonly_rsync_gate.sh /Users/Shared/mango_calls_readonly_rsync_gate.sh
sudo chown root:wheel /Users/Shared/mango_calls_readonly_rsync_gate.sh
sudo chmod 755 /Users/Shared/mango_calls_readonly_rsync_gate.sh
sudo ln -sfn "/Users/dmitrijfabarisov/Projects/Mango analyse/product_data/mango_calls_two_processes/drop" \
  /Users/Shared/mango_calls_drop_ro
sudo chown -h root:wheel /Users/Shared/mango_calls_drop_ro
```

На основном Mac создать отдельный ключ и зафиксировать fingerprint хоста M1:

```bash
M1_HOSTNAME='ИМЯ-ИЛИ-IP-M1'
ssh-keygen -t ed25519 -f "$HOME/.ssh/mango_calls_m1_reader" -C mango-calls-readonly
ssh-keyscan -t ed25519 "$M1_HOSTNAME" > "$HOME/.ssh/mango_calls_m1_known_hosts.candidate"
ssh-keygen -lf "$HOME/.ssh/mango_calls_m1_known_hosts.candidate"
```

Полученный fingerprint вручную сравнить на M1 с
`ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub`. Только после совпадения
переименовать candidate в `~/.ssh/mango_calls_m1_known_hosts` и поставить режим
`0600`.

Публичную часть `~/.ssh/mango_calls_m1_reader.pub` вручную добавить на M1 в
`~/.ssh/authorized_keys` отдельной строкой с префиксом:

```text
restrict,command="/Users/Shared/mango_calls_readonly_rsync_gate.sh /Users/Shared/mango_calls_drop_ro" ssh-ed25519 ... mango-calls-readonly
```

До запуска проверить этим ключом: два файла drop читаются, произвольная команда
и чтение каталога секретов получают отказ. Puller использует именно этот ключ,
отдельный known_hosts, `BatchMode=yes` и `StrictHostKeyChecking=yes`.

## Настройка M1: только Process A

В отдельном файле `~/.mango_secrets/mango_calls_m1_worker.env` находятся Mango,
Tallanto, Google и пути публикации. Пример имён без значений:

```text
MANGO_OFFICE_API_KEY=
MANGO_OFFICE_API_SALT=
MANGO_CALLS_EXPECTED_CODE_SHA=<один подтверждённый SHA для обоих компьютеров>
MANGO_CALLS_DAILY_EXPORT_OUT=/Users/dmitrijfabarisov/Yandex.Disk.localized/Mango Calls Resolve
MANGO_CALLS_GOOGLE_DRIVE_FOLDER_ID=
GOOGLE_APPLICATION_CREDENTIALS=/Users/dmitrijfabarisov/.mango_secrets/google-mango-calls.json
```

Сначала только отрисовать и проверить plist:

```bash
/usr/bin/python3 scripts/install_mango_calls_two_processes_service.py \
  --config ~/.mango_local/mango_calls_two_processes/config.json \
  --env-file ~/.mango_secrets/mango_calls_m1_worker.env \
  --process-a-only --process-a-interval-seconds 1800 \
  --out-dir ~/.mango_local/mango_calls_two_processes/launchd-preview
```

В plist должен быть только `com.mango.calls-process-a` с командой
`process-a-worker`. После ручной проверки убрать старые задания из штатного
каталога, сохранив их в `$SNAP`, и устанавливать без `--out-dir`:

```bash
M1_SNAP="$HOME/.mango_local/mango_calls_cutover/$(date -u +%Y%m%dT%H%M%SZ)_m1"
mkdir -p "$M1_SNAP" && chmod 700 "$M1_SNAP"
printf '%s\n' "$M1_SNAP" > "$HOME/.mango_local/mango_calls_cutover/M1_SNAPSHOT_PATH"
chmod 600 "$HOME/.mango_local/mango_calls_cutover/M1_SNAPSHOT_PATH"
cp -p ~/Library/LaunchAgents/com.mango.calls-process-*.plist "$M1_SNAP/" 2>/dev/null || true
cp -p ~/.mango_local/mango_calls_two_processes/config.json "$M1_SNAP/"
launchctl bootout "gui/$(id -u)/com.mango.calls-process-b" 2>/dev/null || true
launchctl bootout "gui/$(id -u)/com.mango.calls-two-processes" 2>/dev/null || true
mv ~/Library/LaunchAgents/com.mango.calls-process-b.plist "$M1_SNAP/" 2>/dev/null || true
mv ~/Library/LaunchAgents/com.mango.calls-two-processes.plist "$M1_SNAP/" 2>/dev/null || true
/usr/bin/python3 scripts/install_mango_calls_two_processes_service.py \
  --config ~/.mango_local/mango_calls_two_processes/config.json \
  --env-file ~/.mango_secrets/mango_calls_m1_worker.env \
  --process-a-only --process-a-interval-seconds 1800 --install
```

Установщик проверяет настоящую файловую блокировку, а не старый PID в JSON.
Локальный Process B, старое задание или оставшийся конфликтующий plist блокируют
установку.

## Настройка основного Mac: только получение и Process B

Файл `~/.mango_secrets/mango_calls_main_receiver.env`, режим `0600`:

```text
MANGO_CALLS_REMOTE_HOST=dmitrijfabarisov@M1_HOST
MANGO_CALLS_REMOTE_SSH_KEY=/Users/dmitrijfabarisov/.ssh/mango_calls_m1_reader
MANGO_CALLS_REMOTE_KNOWN_HOSTS=/Users/dmitrijfabarisov/.ssh/mango_calls_m1_known_hosts
MANGO_CALLS_EXPECTED_CODE_SHA=<тот же подтверждённый SHA>
MANGO_CALLS_REMOTE_DROP_ROOT=/Users/Shared/mango_calls_drop_ro
MANGO_CALLS_REMOTE_INCOMING_ROOT=/Users/dmitrijfabarisov/.mango_local/mango_calls_remote_incoming
```

Сначала только отрисовать plist:

```bash
/usr/bin/python3 scripts/install_mango_calls_two_processes_service.py \
  --config ~/.mango_local/mango_calls_two_processes/config.json \
  --env-file ~/.mango_secrets/mango_calls_main_receiver.env \
  --process-b-only --process-b-interval-seconds 900 \
  --out-dir ~/.mango_local/mango_calls_two_processes/launchd-preview
```

В plist должен быть только `com.mango.calls-process-b` с командой
`process-b-pull`. Установка блокируется, пока локальный Process A загружен.
После проверки убрать старые задания в `$SNAP` и установить только B в штатный
каталог:

```bash
MAIN_SNAP="$HOME/.mango_local/mango_calls_cutover/$(date -u +%Y%m%dT%H%M%SZ)_main"
mkdir -p "$MAIN_SNAP" && chmod 700 "$MAIN_SNAP"
printf '%s\n' "$MAIN_SNAP" > "$HOME/.mango_local/mango_calls_cutover/MAIN_SNAPSHOT_PATH"
chmod 600 "$HOME/.mango_local/mango_calls_cutover/MAIN_SNAPSHOT_PATH"
cp -p ~/Library/LaunchAgents/com.mango.calls-process-*.plist "$MAIN_SNAP/" 2>/dev/null || true
cp -p ~/.mango_local/mango_calls_two_processes/config.json "$MAIN_SNAP/"
PIPELINE_ROOT="$(/usr/bin/plutil -extract pipeline_root raw -o - \
  ~/.mango_local/mango_calls_two_processes/config.json)"
shasum -a 256 "$PIPELINE_ROOT/drop/"* > "$MAIN_SNAP/drop.sha256"
launchctl bootout "gui/$(id -u)/com.mango.calls-process-a" 2>/dev/null || true
launchctl bootout "gui/$(id -u)/com.mango.calls-two-processes" 2>/dev/null || true
mv ~/Library/LaunchAgents/com.mango.calls-process-a.plist "$MAIN_SNAP/" 2>/dev/null || true
mv ~/Library/LaunchAgents/com.mango.calls-two-processes.plist "$MAIN_SNAP/" 2>/dev/null || true
/usr/bin/python3 scripts/install_mango_calls_two_processes_service.py \
  --config ~/.mango_local/mango_calls_two_processes/config.json \
  --env-file ~/.mango_secrets/mango_calls_main_receiver.env \
  --process-b-only --process-b-interval-seconds 900 --install
```

Каждый цикл основного Mac забирает manifest, затем SQLite, затем manifest ещё
раз. Принимаются только неизменившийся manifest, совпавшие SHA-256 и размер,
`quick_check=ok` и `integrity_check=ok`. DB заменяется первой, manifest последним;
предыдущее поколение хранится одним локальным rollback-hardlink. Только после
успешной приёмки выполняется Process B.

## Первый контрольный цикл

1. На M1 вручную запустить Process A и дождаться готовой ready-базы:
   ```bash
   /bin/zsh scripts/run_mango_calls_process.sh \
     ~/.mango_local/mango_calls_two_processes/config.json \
     ~/.mango_secrets/mango_calls_m1_worker.env process-a-worker
   ```
2. Проверить, что Process B на M1 не запущен.
3. На основном Mac вручную запустить получение и Process B:
   ```bash
   /bin/zsh scripts/run_mango_calls_process.sh \
     ~/.mango_local/mango_calls_two_processes/config.json \
     ~/.mango_secrets/mango_calls_main_receiver.env process-b-pull
   ```
4. Сверить SHA ready-базы на обоих компьютерах и зелёный отчёт Process B.
5. Проверить XLSX, TXT, Google-таблицу и отсутствие файла с именем
   `ПРОВЕРКА — НЕ ИСПОЛЬЗОВАТЬ` для успешно завершённого поколения.
6. Проверить второй неизменный запуск: повторного скачивания итогов и дубля
   Google-таблицы быть не должно.

## Откат

Если первый контрольный цикл не прошёл, на M1 выключить новый Process A и
вернуть только те прежние задания, которые реально были в `M1_SNAP`:

```bash
set -euo pipefail
M1_SNAP="$(cat "$HOME/.mango_local/mango_calls_cutover/M1_SNAPSHOT_PATH")"
test -n "$M1_SNAP" && test -f "$M1_SNAP/config.json"
launchctl bootout "gui/$(id -u)/com.mango.calls-process-a" 2>/dev/null || true
cp -p "$M1_SNAP/config.json" ~/.mango_local/mango_calls_two_processes/config.json
find "$M1_SNAP" -maxdepth 1 -type f -name '*.plist' -print | while IFS= read -r plist; do
  cp -p "$plist" ~/Library/LaunchAgents/
  launchctl bootstrap "gui/$(id -u)" ~/Library/LaunchAgents/"${plist##*/}"
done
```

На основном Mac выключить раздельный Process B, вернуть сохранённые config и
plist, проверить исходный drop и загрузить прежнюю локальную схему:

```bash
set -euo pipefail
MAIN_SNAP="$(cat "$HOME/.mango_local/mango_calls_cutover/MAIN_SNAPSHOT_PATH")"
test -n "$MAIN_SNAP" && test -f "$MAIN_SNAP/config.json" && test -f "$MAIN_SNAP/drop.sha256"
launchctl bootout "gui/$(id -u)/com.mango.calls-process-b" 2>/dev/null || true
cp -p "$MAIN_SNAP/config.json" ~/.mango_local/mango_calls_two_processes/config.json
PIPELINE_ROOT="$(/usr/bin/plutil -extract pipeline_root raw -o - \
  ~/.mango_local/mango_calls_two_processes/config.json)"
(cd "$PIPELINE_ROOT/drop" && shasum -a 256 -c "$MAIN_SNAP/drop.sha256")
find "$MAIN_SNAP" -maxdepth 1 -type f -name '*.plist' -print | while IFS= read -r plist; do
  cp -p "$plist" ~/Library/LaunchAgents/
  launchctl bootstrap "gui/$(id -u)" ~/Library/LaunchAgents/"${plist##*/}"
done
```

Reverse-copy базы с M1 запрещён без отдельного расследования. Старые данные не
удалять до семи последовательных зелёных суточных циклов и решения владельца.

Если принятая удалённая ready-база оказалась логически непригодной, сначала
выключить Process B, затем из чистой подтверждённой ревизии выполнить локальный
обмен текущего и предыдущего поколений:

```bash
PIPELINE_ROOT="$(/usr/bin/plutil -extract pipeline_root raw -o - \
  ~/.mango_local/mango_calls_two_processes/config.json)"
python3 scripts/receive_mango_calls_drop.py \
  --pipeline-root "$PIPELINE_ROOT" \
  --repo-root "$PWD" --expected-code-sha "$(git rev-parse HEAD)" \
  --restore-rollback --execute \
  --confirmation RESTORE_MANGO_CALLS_REMOTE_ROLLBACK
```

Команда сначала полностью проверяет резервную SQLite, сохраняет текущее
поколение как обратный откат, заменяет базу первой и manifest последним.

Аудио новых звонков после переключения остаётся на M1. Process B переносит в
Timeline текст и служебный исходный путь, но этот путь не гарантирует открытие
аудио на основном Mac. Текущая задача РОПа использует TXT-расшифровки; если
понадобится открывать исходное аудио с основного Mac, это отдельный защищённый
read-only механизм, а не повод копировать аудио через Яндекс Диск.

## Этап 10

Этап 10 — семь последовательных суточных циклов на M1 с контролем полноты,
скорости, разделения ролей, отсутствия дублей, публикации Яндекс/Google и
доставки ready-базы в Timeline. Его точная карта приёмки находится в
`tasks/_inbox_codex/2026-07-31_TZ_m1_calls_stage10_pilot.md`.
