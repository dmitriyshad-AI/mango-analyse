# Перенос обработки звонков на M1

> **STOP: это архивная split-топология, а не инструкция запуска fast-service.**
> Каноническая Phase A описана в
> `docs/MANGO_CALLS_TWO_PROCESSES_RUNBOOK.md` и
> `tasks/_running/2026-08-11_TZ_m1_calls_service_fast_value.md`. Из этого
> файла после отдельного допуска Phase C используются только
> блоки безопасного selective transfer/relocation. Команды старой
> схемы с Process B на основном Mac и launchd исполнять запрещено.

Актуализировано: 2026-08-11. Старый SHA внешнего пакета от 2026-08-01
не использовать.

## Порядок без перескоков

1. До слияния read-only проверить, что ни одно старое задание звонков не
   загружено. Если загружено - остановиться и получить отдельное решение
   владельца об остановке до слияния.
2. Закрыть кодовые гейты, влить их в `main`, затем зафиксировать
   `CUTOVER_CODE_SHA`.
3. Подготовить M1 и выполнить только read-only проверки.
4. По отдельному подтверждению остановить старый Process A и напрямую перенести
   данные.
5. Выполнить dry-run на синтетике.
6. По новому подтверждению провести один ручной контрольный цикл.
7. Только после его PASS выполнить `--install`; эта команда сразу запускает
   службу и является live-cutover.
8. Пройти семь полных последовательных московских суток.

## Архивная целевая схема — не исполнять

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

До слияния изменения старого entrypoint выполнить только read-only проверку:

```bash
set -euo pipefail
loaded=0
deprecated_runner='run_mango_calls'"_cycle.sh"
for label in com.mango.calls-two-processes com.mango.calls-process-a com.mango.calls-process-b; do
  if launchctl print "gui/$(id -u)/$label" >/dev/null 2>&1; then
    printf 'STOP: loaded launchd label: %s\n' "$label" >&2
    loaded=1
  fi
done
process_matches="$(ps -axo command= | /usr/bin/awk \
  '/[r]un_mango_calls_(cycle|process)\.sh|[r]un_mango_calls_pipeline\.py/ {print}')"
if [[ -n "$process_matches" ]]; then
  printf 'STOP: running calls process found\n' >&2
  loaded=1
fi
cron_matches="$(crontab -l 2>/dev/null | /usr/bin/awk \
  '/[r]un_mango_calls_|[m]ango_calls_pipeline/ {print}' || true)"
if [[ -n "$cron_matches" ]]; then
  printf 'STOP: calls scheduler found in crontab\n' >&2
  loaded=1
fi
for plist in "$HOME"/Library/LaunchAgents/com.mango.calls-*.plist; do
  [[ -e "$plist" ]] || continue
  test -f "$plist" && test ! -L "$plist"
  runner="$(/usr/bin/plutil -extract ProgramArguments.1 raw -o - "$plist")"
  config="$(/usr/bin/plutil -extract ProgramArguments.2 raw -o - "$plist")"
  test -f "$runner" && test -f "$config"
  if [[ "$runner" == *"$deprecated_runner" ]]; then
    printf 'STOP: deprecated calls runner in plist: %s\n' "${plist##*/}" >&2
    loaded=1
  fi
done
(( loaded == 0 ))
```

При ненулевом результате не сливать ветку и не останавливать службу без
отдельного подтверждения владельца. В owner-only доказательстве сохранить время,
проверенные метки, число найденных процессов/cron-заданий и безопасные имена
runner/config без значений env.

## Проверка обоих компьютеров

1. На обоих компьютерах одна и та же чистая ревизия Git:

   ```bash
   set -euo pipefail
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
4. На M1 установлен клиент Яндекс Диска, синхронизация подтверждена круговым
   тестом, а в целевой папке есть маркер `.mango_calls_yandex_target`. Google
   необязателен: оба его параметра либо пусты, либо полностью настроены; ключ
   хранится отдельно с режимом `0600`, папка непубличная.
5. Основной Mac фиксирует ключ хоста M1 в отдельном `known_hosts`. M1 не имеет
   SSH-ключа к основному Mac. Для чтения ready-drop используется отдельный ключ
   текущего пользователя M1 с принудительной read-only командой из раздела ниже.

## Первичная передача рабочего состояния

Передача выполняется один раз с основного Mac напрямую на M1 по SSH, только
после штатного завершения Process A. Старый источник пока находится внутри
репозитория, но M1 получает данные в закрытый каталог `~/.mango_local`, вне Git
и облачных папок. Поэтому кодовая Фаза 0 обязана сначала реализовать и проверить
на синтетике одноразовое исправление сохранённых абсолютных путей.

Сначала через фактически загруженный plist/`launchctl print` найти абсолютный
путь config действующего Process A. Не угадывать его по документации. Подставить
его вместо placeholder и сохранить доказательства для отката:

```bash
set -euo pipefail
OWNER_LOCAL="$HOME/.mango_local"
test ! -L "$OWNER_LOCAL"
mkdir -p "$OWNER_LOCAL"
test -d "$OWNER_LOCAL" && test "$(stat -f '%u' "$OWNER_LOCAL")" = "$(id -u)"
chmod 700 "$OWNER_LOCAL"
test "$(stat -f '%u:%Lp' "$OWNER_LOCAL")" = "$(id -u):700"
SOURCE_CONFIG='<ФАКТИЧЕСКИЙ АБСОЛЮТНЫЙ ПУТЬ CONFIG ИЗ ЗАГРУЖЕННОГО PROCESS A>'
[[ "$SOURCE_CONFIG" == /* && "$SOURCE_CONFIG" != *'<'* ]]
test -f "$SOURCE_CONFIG" && test ! -L "$SOURCE_CONFIG"
SNAP="$HOME/.mango_local/mango_calls_cutover/$(date -u +%Y%m%dT%H%M%SZ)"
SOURCE_PIPELINE="$(/usr/bin/plutil -extract pipeline_root raw -o - \
  "$SOURCE_CONFIG")"
[[ "$SOURCE_PIPELINE" == /* && "$SOURCE_PIPELINE" != / ]]
test -d "$SOURCE_PIPELINE/capture" && test -d "$SOURCE_PIPELINE/working" && test -d "$SOURCE_PIPELINE/drop"
mkdir -p "$SNAP" && chmod 700 "$SNAP"
printf '%s\n' "$SNAP" > "$HOME/.mango_local/mango_calls_cutover/MAIN_SOURCE_SNAPSHOT_PATH"
chmod 600 "$HOME/.mango_local/mango_calls_cutover/MAIN_SOURCE_SNAPSHOT_PATH"
: > "$SNAP/active_labels.txt"
for label in com.mango.calls-process-a com.mango.calls-process-b com.mango.calls-two-processes; do
  if launchctl print "gui/$(id -u)/$label" > "$SNAP/$label.before.txt" 2>&1; then
    printf '%s\n' "$label" >> "$SNAP/active_labels.txt"
  fi
done
chmod 600 "$SNAP/active_labels.txt"
cp -p ~/Library/LaunchAgents/com.mango.calls-*.plist "$SNAP/" 2>/dev/null || true
while IFS= read -r label; do
  test -f "$SNAP/$label.plist"
done < "$SNAP/active_labels.txt"
cp -p "$SOURCE_CONFIG" "$SNAP/config.json"
printf '%s\n' "$SOURCE_CONFIG" > "$SNAP/SOURCE_CONFIG_PATH"
chmod 600 "$SNAP/SOURCE_CONFIG_PATH"
shasum -a 256 \
  "$SOURCE_PIPELINE/drop/"* \
  > "$SNAP/drop.sha256"
```

После снимка выгрузить оба возможных задания Process A и дождаться освобождения
настоящей файловой блокировки. Перенос до этого запрещён:

```bash
set -euo pipefail
SNAP="$(cat "$HOME/.mango_local/mango_calls_cutover/MAIN_SOURCE_SNAPSHOT_PATH")"
test -d "$SNAP" && test -f "$SNAP/config.json"
SOURCE_PIPELINE="$(/usr/bin/plutil -extract pipeline_root raw -o - \
  "$SNAP/config.json")"
[[ "$SOURCE_PIPELINE" == /* && "$SOURCE_PIPELINE" != / ]]
test -d "$SOURCE_PIPELINE/capture" && test -d "$SOURCE_PIPELINE/working" && test -d "$SOURCE_PIPELINE/drop"
launchctl bootout "gui/$(id -u)/com.mango.calls-process-a" 2>/dev/null || true
launchctl bootout "gui/$(id -u)/com.mango.calls-process-b" 2>/dev/null || true
launchctl bootout "gui/$(id -u)/com.mango.calls-two-processes" 2>/dev/null || true
! launchctl print "gui/$(id -u)/com.mango.calls-process-a" >/dev/null 2>&1
! launchctl print "gui/$(id -u)/com.mango.calls-process-b" >/dev/null 2>&1
! launchctl print "gui/$(id -u)/com.mango.calls-two-processes" >/dev/null 2>&1
python3 - "$SOURCE_PIPELINE" <<'PY'
import fcntl
import sys
from pathlib import Path
root = Path(sys.argv[1])
for name in ('process_a.lock', 'process_b.lock'):
    path = root / 'locks' / name
    if not path.exists():
        continue
    with path.open('rb') as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
PY
SOURCE_INVENTORY="$SNAP/source_inventory.json"
python3 scripts/relocate_mango_calls_pipeline.py \
  --inventory-root "$SOURCE_PIPELINE" --inventory-out "$SOURCE_INVENTORY"
chmod 600 "$SOURCE_INVENTORY"
python3 scripts/relocate_mango_calls_pipeline.py \
  --verify-inventory "$SOURCE_INVENTORY" --inventory-root "$SOURCE_PIPELINE"
assert_existing_single_link_file() {
  local path="$1"
  test -f "$path"
  test ! -L "$path"
  test "$(stat -f '%u' "$path")" = "$(id -u)"
  test "$(stat -f '%l' "$path")" = "1"
}
assert_existing_single_link_database() {
  local database="$1"
  assert_existing_single_link_file "$database"
  test -s "$database"
}
SOURCE_WORKING_DB="$SOURCE_PIPELINE/working/mango_calls_pipeline.sqlite"
assert_existing_single_link_database "$SOURCE_WORKING_DB"
SOURCE_WORKING_WAL="${SOURCE_WORKING_DB}-wal"
if [[ -e "$SOURCE_WORKING_WAL" ]]; then
  assert_existing_single_link_file "$SOURCE_WORKING_WAL"
  test ! -s "$SOURCE_WORKING_WAL"
fi
SOURCE_WORKING_SHM="${SOURCE_WORKING_DB}-shm"
if [[ -e "$SOURCE_WORKING_SHM" ]]; then
  assert_existing_single_link_file "$SOURCE_WORKING_SHM"
  test "$(stat -f '%z' "$SOURCE_WORKING_SHM")" = "32768"
fi
```

Если lock занят, WAL непуст или storage небезопасен, остановиться. Не убивать
процесс и не копировать работающую базу. Полный inventory выполняется до любой
передачи и отклоняет ссылки, специальные файлы и внешние hardlink во всём
дереве, включая SQLite sidecar. Source после inventory остаётся байт-в-байт
неизменным: вручную делать checkpoint или перепечатывать ready-manifest
запрещено. Непустой WAL требует отдельного разрешения Дмитрия на штатный
republish владельцем ready DB — Process A — в отдельно согласованном режиме без
capture, ASR, Resolve и Analyze; после него все три задания снова останавливаются,
а этот блок и inventory выполняются заново.

Следующий блок и все дальнейшие блоки с
`scripts/relocate_mango_calls_pipeline.py` доступны только после завершения
кодовой Фазы 0 и появления проверенного скрипта. До этого остановиться.

Затем построить выборочный контракт передачи. Он включает все БД, manifest,
cursor, recovery ledger, SQLite sidecar, неготовое/multi/unreferenced аудио и
все прочие файлы. Исключается только аудио уникального звонка, строго готового
одновременно в working и sealed ready DB. Это особенно важно после падения
между скачиванием и append: неизвестный orphan переносится по умолчанию.

Встроенный macOS `openrsync` проверяется живым dry-run команды `--from0`, а не
по тексту `--help`. `files-from` имеет NUL-разделители; превращать его в строки
через shell запрещено.

```bash
set -euo pipefail
M1_USER='dmitriy'
M1_HOST="$M1_USER@ИМЯ-ИЛИ-IP-M1"
[[ "$M1_HOST" != *'ИМЯ-ИЛИ-IP-M1'* ]]
SNAP="$(cat "$HOME/.mango_local/mango_calls_cutover/MAIN_SOURCE_SNAPSHOT_PATH")"
test -d "$SNAP" && test -f "$SNAP/config.json"
SOURCE_PIPELINE="$(/usr/bin/plutil -extract pipeline_root raw -o - \
  "$SNAP/config.json")"
[[ "$SOURCE_PIPELINE" == /* && "$SOURCE_PIPELINE" != / ]]
test -d "$SOURCE_PIPELINE/capture" && test -d "$SOURCE_PIPELINE/working" && test -d "$SOURCE_PIPELINE/drop"
GENERATION="$(date -u +%Y%m%dT%H%M%SZ)"
printf '%s\n' "$GENERATION" > "$SNAP/GENERATION_ID"
SOURCE_INVENTORY="$SNAP/selective_source_inventory.json"
FILES_FROM="$SNAP/selective_files_from.nul"
python3 scripts/relocate_mango_calls_pipeline.py \
  --selective-inventory-root "$SOURCE_PIPELINE" \
  --inventory-out "$SOURCE_INVENTORY" \
  --files-from-out "$FILES_FROM"
test -s "$SOURCE_INVENTORY" && test -s "$FILES_FROM"
test "$(stat -f '%Lp' "$SOURCE_INVENTORY")" = 600
test "$(stat -f '%Lp' "$FILES_FROM")" = 600
python3 scripts/relocate_mango_calls_pipeline.py \
  --verify-selective-source "$SOURCE_INVENTORY" \
  --selective-inventory-root "$SOURCE_PIPELINE" \
  --files-from-out "$FILES_FROM"
SOURCE_PREFLIGHT_TARGET="$HOME/.mango_local/mango_calls_source_preflight_only"
test ! -e "$SOURCE_PREFLIGHT_TARGET"
python3 scripts/relocate_mango_calls_pipeline.py \
  --pipeline-root "$SOURCE_PIPELINE" --old-root "$SOURCE_PIPELINE" \
  --new-root "$SOURCE_PREFLIGHT_TARGET" \
  --source-inventory "$SOURCE_INVENTORY" --dry-run
test ! -e "$SOURCE_PREFLIGHT_TARGET"
ssh "$M1_HOST" 'set -euo pipefail; OWNER_LOCAL="$HOME/.mango_local"; test ! -L "$OWNER_LOCAL"; mkdir -p "$OWNER_LOCAL"; test -d "$OWNER_LOCAL"; test "$(stat -f %u "$OWNER_LOCAL")" = "$(id -u)"; chmod 700 "$OWNER_LOCAL"; test "$(stat -f %u:%Lp "$OWNER_LOCAL")" = "$(id -u):700"'
ssh "$M1_HOST" "set -euo pipefail; umask 077; TARGET=~/.mango_local/mango_calls_transfers/$GENERATION; test ! -e \"\$TARGET\"; mkdir -p \"\$TARGET\"; chmod 700 \"\$TARGET\""
/usr/bin/rsync -aH --relative --from0 --files-from="$FILES_FROM" \
  "$SOURCE_PIPELINE/" \
  "$M1_HOST:~/.mango_local/mango_calls_transfers/$GENERATION/"
scp "$SOURCE_INVENTORY" \
  "$M1_HOST:~/.mango_local/mango_calls_transfers/$GENERATION.source_inventory.json"
scp "$FILES_FROM" \
  "$M1_HOST:~/.mango_local/mango_calls_transfers/$GENERATION.files_from.nul"
ssh "$M1_HOST" "chmod 600 ~/.mango_local/mango_calls_transfers/$GENERATION.source_inventory.json ~/.mango_local/mango_calls_transfers/$GENERATION.files_from.nul"
python3 scripts/relocate_mango_calls_pipeline.py \
  --verify-selective-source "$SOURCE_INVENTORY" \
  --selective-inventory-root "$SOURCE_PIPELINE" \
  --files-from-out "$FILES_FROM"
RSYNC_DIFF="$SNAP/rsync_second_pass.txt"
/usr/bin/rsync -aHn --relative --from0 --files-from="$FILES_FROM" --itemize-changes \
  "$SOURCE_PIPELINE/" \
  "$M1_HOST:~/.mango_local/mango_calls_transfers/$GENERATION/" \
  > "$RSYNC_DIFF"
chmod 600 "$RSYNC_DIFF"
test ! -s "$RSYNC_DIFF"
```

Последняя команда обязана вернуть пустую строку: это доказательство, что повтор
того же выборочного контракта не видит изменений. Проверка source выполняется и
до, и после rsync. `--delete` здесь намеренно запрещён: target является точным
подмножеством, а ошибка в списке не должна удалять уже переданные данные.
Встроенный `openrsync` округляет `mtime` до секунд, поэтому межмашинная сверка
использует путь, размер, SHA-256 и mode; полные owner-only inventory всё равно
сохраняют исходные `mtime_ns`, а локальный resume проверяет их строго.
Предварительный `--dry-run` выше ничего не создаёт и заранее ловит проблемы
source runtime; новый путь M1 он не подтверждает, поэтому точный dry-run на M1
после передачи остаётся обязательным.
Process A на основном Mac остаётся выключенным. Затем на M1 указать тот же
`GENERATION` и проверить полный манифест до изменения путей:

```bash
set -euo pipefail
umask 077
OWNER_LOCAL="$HOME/.mango_local"
test ! -L "$OWNER_LOCAL"
test -d "$OWNER_LOCAL"
test "$(stat -f '%u:%Lp' "$OWNER_LOCAL")" = "$(id -u):700"
GENERATION='<GENERATION_ID из снимка основного Mac>'
[[ "$GENERATION" != *'<'* ]]
PIPELINE_TRANSFER="$HOME/.mango_local/mango_calls_transfers/$GENERATION"
PIPELINE="$HOME/.mango_local/mango_calls_two_processes"
test -d "$PIPELINE_TRANSFER"
INVENTORY="$PIPELINE_TRANSFER.source_inventory.json"
FILES_FROM="$PIPELINE_TRANSFER.files_from.nul"
test -f "$INVENTORY" && test ! -L "$INVENTORY"
test -f "$FILES_FROM" && test ! -L "$FILES_FROM"
chmod 600 "$INVENTORY" "$FILES_FROM"
OLD_PIPELINE="$(python3 - "$INVENTORY" <<'PY'
import json, sys
value = json.load(open(sys.argv[1], encoding='utf-8')).get('source_root')
assert isinstance(value, str) and value.startswith('/') and value != '/'
print(value)
PY
)"
test -z "$(find "$PIPELINE_TRANSFER" -type l -print -quit)"
python3 scripts/relocate_mango_calls_pipeline.py \
  --verify-inventory "$INVENTORY" --inventory-root "$PIPELINE_TRANSFER"
python3 scripts/relocate_mango_calls_pipeline.py \
  --pipeline-root "$PIPELINE_TRANSFER" --old-root "$OLD_PIPELINE" --new-root "$PIPELINE" \
  --source-inventory "$INVENTORY" --dry-run
CONFIRM_MANGO_CALLS_RELOCATION=RELOCATE_MANGO_CALLS_PIPELINE \
  python3 scripts/relocate_mango_calls_pipeline.py \
  --pipeline-root "$PIPELINE_TRANSFER" --old-root "$OLD_PIPELINE" --new-root "$PIPELINE" \
  --source-inventory "$INVENTORY" --execute
REPEAT_RESULT="$(CONFIRM_MANGO_CALLS_RELOCATION=RELOCATE_MANGO_CALLS_PIPELINE \
  python3 scripts/relocate_mango_calls_pipeline.py \
  --pipeline-root "$PIPELINE_TRANSFER" --old-root "$OLD_PIPELINE" --new-root "$PIPELINE" \
  --source-inventory "$INVENTORY" --execute)"
python3 - "$REPEAT_RESULT" <<'PY'
import json, sys
result = json.loads(sys.argv[1])
assert result.get('status') == 'already_relocated' and result.get('changes') == 0
PY
test -z "$(find "$PIPELINE_TRANSFER" \( -type d ! -perm 700 -o -type f ! -perm 600 \) -print)"
if [[ -e "$PIPELINE" ]]; then
  test -d "$PIPELINE" && test ! -L "$PIPELINE"
  test -z "$(find "$PIPELINE" -mindepth 1 -print -quit)"
  rmdir "$PIPELINE"
fi
mv "$PIPELINE_TRANSFER" "$PIPELINE"
M1_BOOTSTRAP_SINCE='<ПОДТВЕРЖДЁННАЯ UTC-ДАТА НАЧАЛА ПЕРВОГО ОКНА>'
[[ "$M1_BOOTSTRAP_SINCE" != *'<'* ]]
EXPECTED_CODE_SHA="$(git rev-parse HEAD)"
[[ "$EXPECTED_CODE_SHA" =~ ^[0-9a-f]{40}$ ]]
OWNER_EMAIL='<EMAIL ВЛАДЕЛЬЦА ЗАКРЫТОЙ GOOGLE-ТАБЛИЦЫ>'
ROP_EMAIL='<EMAIL РОПА>'
[[ "$OWNER_EMAIL" != *'<'* && "$ROP_EMAIL" != *'<'* ]]
python3 - docs/m1_calls_handoff_20260801/config.m1.example.json \
  "$PIPELINE/config.json.tmp" "$HOME" "$M1_BOOTSTRAP_SINCE" \
  "$EXPECTED_CODE_SHA" "$OWNER_EMAIL" "$ROP_EMAIL" <<'PY'
import json, sys
source, target, home, since, sha, owner_email, rop_email = sys.argv[1:]
data = json.load(open(source, encoding='utf-8'))
replacements = {
    '<HOME>': home,
    '<CUTOVER_BOOTSTRAP_SINCE_ISO8601>': since,
    '<EXPECTED_CODE_SHA>': sha,
    '<OWNER_EMAIL>': owner_email,
    '<ROP_EMAIL>': rop_email,
}
def render(value):
    if isinstance(value, str):
        for old, new in replacements.items():
            value = value.replace(old, new)
        return value
    if isinstance(value, list):
        return [render(item) for item in value]
    if isinstance(value, dict):
        return {key: render(item) for key, item in value.items()}
    return value
data = render(data)
assert '<' not in json.dumps(data, ensure_ascii=False)
with open(target, 'x', encoding='utf-8') as stream:
    json.dump(data, stream, ensure_ascii=False, indent=2)
    stream.write('\n')
PY
chmod 600 "$PIPELINE/config.json.tmp"
mv "$PIPELINE/config.json.tmp" "$PIPELINE/config.json"
test "$(/usr/bin/plutil -extract pipeline_root raw -o - "$PIPELINE/config.json")" = "$PIPELINE"
python3 scripts/relocate_mango_calls_pipeline.py --check-sqlite \
  "$PIPELINE/working/mango_calls_pipeline.sqlite" \
  "$PIPELINE/drop/mango_calls_ready.sqlite"
```

Скрипт исправляет только абсолютные пути под старым корнем, проверяет полный
source/target-манифест и наличие каждого перенесённого файла, затем пишет
полные owner-only inventory до/после в
`$HOME/.mango_local/mango_calls_relocation_state/`. Имена файлов могут содержать
персональные данные: эти inventory нельзя класть в Git или audit pack; наружу
выносится только сводка без имён. Скрипт принимает полный inventory только под
реальным `$HOME/.mango_local` и отклоняет Git, cloud-sync и символические ссылки.
Любой путь вне старого корня, отсутствующий
или лишний файл, непустой второй `rsync` либо повтор с изменениями означает STOP.
До первого durable plan даже путь под новым корнем означает STOP: смесь старых
и новых путей разрешена только при возобновлении уже записанного plan. State и
staging закрепляются открытыми дескрипторами; staged SQLite строится только там
и не использует `TMPDIR`, Git, Яндекс.Диск или другой cloud-sync каталог.
Внешний hardlink блокирует смену прав до изменения inode, внутренние hardlink
проверяются и сохраняются.
После него `stat -f '%Su:%Lp %N'` должен показать текущего пользователя и `700` для
всех runtime-каталогов, `600` для всех обычных файлов. Оба результата SQLite должны
быть `ok`. Это требование относилось к архивной split-схеме. Fast-service
вызывает demand-only Process B внутри тяжёлого pipeline на M1 и пишет
только в отдельную локальную Customer Timeline staging. Production Timeline и
основной Mac не изменяются.

### Отдельный SSH-ключ только на чтение

Чтобы не создавать и не настраивать системного пользователя вслепую, основной
Mac использует отдельный SSH-ключ. На M1 этот ключ разрешён только с
принудительной read-only командой: произвольная команда и доступ к секретам
через него невозможны. В `/Users/Shared` создаются root-owned ссылка и копия
проверяющего скрипта:

```bash
set -euo pipefail
sudo cp scripts/mango_calls_readonly_rsync_gate.sh /Users/Shared/mango_calls_readonly_rsync_gate.sh
sudo chown root:wheel /Users/Shared/mango_calls_readonly_rsync_gate.sh
sudo chmod 755 /Users/Shared/mango_calls_readonly_rsync_gate.sh
sudo ln -sfn "$HOME/.mango_local/mango_calls_two_processes/drop" \
  /Users/Shared/mango_calls_drop_ro
sudo chown -h root:wheel /Users/Shared/mango_calls_drop_ro
```

На основном Mac создать отдельный ключ и зафиксировать fingerprint хоста M1:

```bash
set -euo pipefail
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

### Передача секретных env без дублей

На основном Mac передать только исходные env напрямую:

```bash
set -euo pipefail
M1_USER='dmitriy'
M1_HOST="$M1_USER@ИМЯ-ИЛИ-IP-M1"
ssh "$M1_HOST" 'umask 077; mkdir -p ~/.mango_secrets; chmod 700 ~/.mango_secrets'
scp ~/.mango_secrets/mango_office.env "$M1_HOST":~/.mango_secrets/mango_office.env
scp ~/.mango_secrets/tallanto_readonly.env "$M1_HOST":~/.mango_secrets/tallanto_readonly.env
ssh "$M1_HOST" 'chmod 600 ~/.mango_secrets/*.env'
```

На M1 собрать worker-env без повторения Mango-ключей:

```bash
set -euo pipefail
cp ~/.mango_secrets/mango_office.env ~/.mango_secrets/mango_calls_m1_worker.env
sed -n '5,$p' docs/m1_calls_handoff_20260801/mango_calls_m1_worker.env.example \
  | sed "s|<HOME>|$HOME|g" >> ~/.mango_secrets/mango_calls_m1_worker.env
chmod 600 ~/.mango_secrets/mango_calls_m1_worker.env
! grep -q '<HOME>' ~/.mango_secrets/mango_calls_m1_worker.env
```

Затем заменить пустые значения и оставшиеся `<PLACEHOLDER>` локальным редактором.
После замены выполнить, не выводя значения в терминал:

```bash
set -euo pipefail
! grep -q '<' ~/.mango_secrets/mango_calls_m1_worker.env
python3 scripts/mango_calls_env.py ~/.mango_secrets/mango_calls_m1_worker.env
```

Парсер обязан завершиться без ошибки.

Свежую CSV Tallanto передать отдельно. Не использовать жёстко зашитый старый
файл от 2026-06-20:

```bash
set -euo pipefail
M1_USER='dmitriy'
M1_HOST="$M1_USER@ИМЯ-ИЛИ-IP-M1"
[[ "$M1_HOST" != *'ИМЯ-ИЛИ-IP-M1'* ]]
TALLANTO_CONTACTS_SOURCE='/АБСОЛЮТНЫЙ/ПУТЬ/К/СВЕЖЕЙ/Contacts.csv'
test -s "$TALLANTO_CONTACTS_SOURCE"
ssh "$M1_HOST" 'umask 077; mkdir -p ~/.mango_local/tallanto; chmod 700 ~/.mango_local/tallanto'
scp -p "$TALLANTO_CONTACTS_SOURCE" "$M1_HOST":~/.mango_local/tallanto/Contacts_current.csv
ssh "$M1_HOST" 'chmod 600 ~/.mango_local/tallanto/Contacts_current.csv'
```

Дата `MANGO_CALLS_TALLANTO_SNAPSHOT_AS_OF` должна соответствовать содержимому,
а на момент cutover снимок должен быть не старше 24 часов. Обновление после
запуска и проверка возраста реализуются в кодовой Фазе 0.

В отдельном файле `~/.mango_secrets/mango_calls_m1_worker.env` находятся только
Mango, read-only Tallanto и локальные пути worker. Google/Яндекс write-параметры
в Phase A туда не добавляются:

```text
MANGO_OFFICE_API_KEY=
MANGO_OFFICE_API_SALT=
MANGO_CALLS_EXPECTED_CODE_SHA=<один подтверждённый SHA для обоих компьютеров>
MANGO_CALLS_PIPELINE_ROOT=<HOME>/.mango_local/mango_calls_two_processes
MANGO_CALLS_TALLANTO_EXPORT=<HOME>/.mango_local/tallanto/Contacts_current.csv
```

Следующий блок относится только к будущей фазе внешней публикации и без
отдельного разрешения Дмитрия не выполняется. Тогда маркер создаётся только
внутри уже установленной и проверенной папки Яндекс Диска:

```bash
set -euo pipefail
YANDEX="$HOME/Yandex.Disk.localized/Mango Calls Resolve"
test -d "$YANDEX" && test -w "$YANDEX" && test ! -L "$YANDEX"
printf 'mango-calls-yandex-v1\n' > "$YANDEX/.mango_calls_yandex_target"
chmod 600 "$YANDEX/.mango_calls_yandex_target"
```

Маркер не доказывает синхронизацию. Выполнить ручной круговой тест безопасного
файла с другого устройства и записать обезличенный результат в owner-only
`phase1_yandex_roundtrip.json`; без него Фаза 1 не пройдена.

Сначала только отрисовать и проверить plist:

```bash
set -euo pipefail
/usr/bin/python3 scripts/install_mango_calls_two_processes_service.py \
  --config ~/.mango_local/mango_calls_two_processes/config.json \
  --env-file ~/.mango_secrets/mango_calls_m1_worker.env \
  --fast-service \
  --out-dir ~/.mango_local/mango_calls_two_processes/launchd-preview
```

Комплект содержит capture `00/15/30/45`, pipeline `07/37`, watchdog
`12/27/42/57`, demand-only Process B и локальные задачи 06:00/07:00/08:00,
08:30, 08:50. Ни один plist не содержит Google/Yandex `--execute`. На этом
этапе `--install` запрещён. Следующий блок относится к будущему отдельно
разрешённому cutover; в Phase A его не выполнять:

```bash
set -euo pipefail
M1_SNAP="$HOME/.mango_local/mango_calls_cutover/$(date -u +%Y%m%dT%H%M%SZ)_m1"
mkdir -p "$M1_SNAP" && chmod 700 "$M1_SNAP"
printf '%s\n' "$M1_SNAP" > "$HOME/.mango_local/mango_calls_cutover/M1_SNAPSHOT_PATH"
chmod 600 "$HOME/.mango_local/mango_calls_cutover/M1_SNAPSHOT_PATH"
cp -p ~/Library/LaunchAgents/com.mango.calls-*.plist "$M1_SNAP/" 2>/dev/null || true
cp -p ~/.mango_local/mango_calls_two_processes/config.json "$M1_SNAP/"
: > "$M1_SNAP/active_labels.txt"
for label in com.mango.calls-process-a com.mango.calls-process-b com.mango.calls-two-processes; do
  launchctl print "gui/$(id -u)/$label" >/dev/null 2>&1 \
    && printf '%s\n' "$label" >> "$M1_SNAP/active_labels.txt" || true
done
chmod 600 "$M1_SNAP/active_labels.txt"
while IFS= read -r label; do
  test -f "$M1_SNAP/$label.plist"
done < "$M1_SNAP/active_labels.txt"
launchctl bootout "gui/$(id -u)/com.mango.calls-process-b" 2>/dev/null || true
launchctl bootout "gui/$(id -u)/com.mango.calls-two-processes" 2>/dev/null || true
mv ~/Library/LaunchAgents/com.mango.calls-process-b.plist "$M1_SNAP/" 2>/dev/null || true
mv ~/Library/LaunchAgents/com.mango.calls-two-processes.plist "$M1_SNAP/" 2>/dev/null || true
```

Установщик проверяет настоящую файловую блокировку, а не старый PID в JSON.
Локальный Process B, старое задание или оставшийся конфликтующий plist блокируют
установку.

## Архив: основной Mac и Process B — не исполнять

Весь этот раздел сохранён только как история прежнего плана. В fast-service
основной Mac не получает ready DB и не запускает Process B.

Сначала создать отдельную локальную копию runtime для Process B. Старый
внутрирепозиторный источник и его config не изменяются: это основа безопасного
отката. Команда допустима только после успешной проверки копии на M1 и при всё
ещё выключенном старом Process A:

```bash
set -euo pipefail
umask 077
OWNER_LOCAL="$HOME/.mango_local"
test ! -L "$OWNER_LOCAL"
mkdir -p "$OWNER_LOCAL"
test -d "$OWNER_LOCAL" && test "$(stat -f '%u' "$OWNER_LOCAL")" = "$(id -u)"
chmod 700 "$OWNER_LOCAL"
test "$(stat -f '%u:%Lp' "$OWNER_LOCAL")" = "$(id -u):700"
SOURCE_SNAP="$(cat "$HOME/.mango_local/mango_calls_cutover/MAIN_SOURCE_SNAPSHOT_PATH")"
test -d "$SOURCE_SNAP" && test -f "$SOURCE_SNAP/config.json"
OLD_PIPELINE="$(/usr/bin/plutil -extract pipeline_root raw -o - "$SOURCE_SNAP/config.json")"
[[ "$OLD_PIPELINE" == /* && "$OLD_PIPELINE" != / ]]
test -d "$OLD_PIPELINE/capture" && test -d "$OLD_PIPELINE/working" && test -d "$OLD_PIPELINE/drop"
MAIN_PIPELINE="$HOME/.mango_local/mango_calls_two_processes"
MAIN_GENERATION="$(date -u +%Y%m%dT%H%M%SZ)"
MAIN_TRANSFER="$HOME/.mango_local/mango_calls_main_generations/$MAIN_GENERATION"
test ! -e "$MAIN_PIPELINE" && test ! -e "$MAIN_TRANSFER"
mkdir -p "$MAIN_TRANSFER" && chmod 700 "$MAIN_TRANSFER"
python3 scripts/relocate_mango_calls_pipeline.py \
  --inventory-root "$OLD_PIPELINE" --inventory-out "$SOURCE_SNAP/main_source_inventory.json"
chmod 600 "$SOURCE_SNAP/main_source_inventory.json"
python3 scripts/relocate_mango_calls_pipeline.py \
  --pipeline-root "$OLD_PIPELINE" --old-root "$OLD_PIPELINE" --new-root "$MAIN_PIPELINE" \
  --source-inventory "$SOURCE_SNAP/main_source_inventory.json" --dry-run
/usr/bin/rsync -aH "$OLD_PIPELINE/" "$MAIN_TRANSFER/"
MAIN_RSYNC_DIFF="$SOURCE_SNAP/main_rsync_second_pass.txt"
/usr/bin/rsync -aHn --delete --itemize-changes \
  "$OLD_PIPELINE/" "$MAIN_TRANSFER/" > "$MAIN_RSYNC_DIFF"
chmod 600 "$MAIN_RSYNC_DIFF" && test ! -s "$MAIN_RSYNC_DIFF"
python3 scripts/relocate_mango_calls_pipeline.py \
  --verify-inventory "$SOURCE_SNAP/main_source_inventory.json" --inventory-root "$MAIN_TRANSFER"
test -z "$(find "$MAIN_TRANSFER" -type l -print -quit)"
python3 scripts/relocate_mango_calls_pipeline.py \
  --pipeline-root "$MAIN_TRANSFER" --old-root "$OLD_PIPELINE" --new-root "$MAIN_PIPELINE" \
  --source-inventory "$SOURCE_SNAP/main_source_inventory.json" --dry-run
CONFIRM_MANGO_CALLS_RELOCATION=RELOCATE_MANGO_CALLS_PIPELINE \
  python3 scripts/relocate_mango_calls_pipeline.py \
  --pipeline-root "$MAIN_TRANSFER" --old-root "$OLD_PIPELINE" --new-root "$MAIN_PIPELINE" \
  --source-inventory "$SOURCE_SNAP/main_source_inventory.json" --execute
MAIN_REPEAT_RESULT="$(CONFIRM_MANGO_CALLS_RELOCATION=RELOCATE_MANGO_CALLS_PIPELINE \
  python3 scripts/relocate_mango_calls_pipeline.py \
  --pipeline-root "$MAIN_TRANSFER" --old-root "$OLD_PIPELINE" --new-root "$MAIN_PIPELINE" \
  --source-inventory "$SOURCE_SNAP/main_source_inventory.json" --execute)"
python3 - "$MAIN_REPEAT_RESULT" <<'PY'
import json, sys
result = json.loads(sys.argv[1])
assert result.get('status') == 'already_relocated' and result.get('changes') == 0
PY
cp -p "$SOURCE_SNAP/config.json" "$MAIN_TRANSFER/config.json.tmp"
/usr/bin/plutil -replace pipeline_root -string "$MAIN_PIPELINE" "$MAIN_TRANSFER/config.json.tmp"
chmod 600 "$MAIN_TRANSFER/config.json.tmp"
mv "$MAIN_TRANSFER/config.json.tmp" "$MAIN_TRANSFER/config.json"
test -z "$(find "$MAIN_TRANSFER" \( -type d ! -perm 700 -o -type f ! -perm 600 \) -print)"
python3 scripts/relocate_mango_calls_pipeline.py \
  --inventory-root "$MAIN_TRANSFER" --inventory-out "$SOURCE_SNAP/main_final_inventory.json"
chmod 600 "$SOURCE_SNAP/main_final_inventory.json"
mv "$MAIN_TRANSFER" "$MAIN_PIPELINE"
python3 scripts/relocate_mango_calls_pipeline.py --check-sqlite \
  "$MAIN_PIPELINE/working/mango_calls_pipeline.sqlite" \
  "$MAIN_PIPELINE/drop/mango_calls_ready.sqlite"
```

Оба результата SQLite должны быть `ok`, второй `rsync` пустым, а старый
`OLD_PIPELINE` должен остаться на месте без изменений. Иначе остановиться.

Файл `~/.mango_secrets/mango_calls_main_receiver.env`, режим `0600`:

```text
MANGO_CALLS_REMOTE_HOST=dmitriy@M1_HOST
MANGO_CALLS_REMOTE_SSH_KEY=<MAIN_HOME>/.ssh/mango_calls_m1_reader
MANGO_CALLS_REMOTE_KNOWN_HOSTS=<MAIN_HOME>/.ssh/mango_calls_m1_known_hosts
MANGO_CALLS_EXPECTED_CODE_SHA=<тот же подтверждённый SHA>
MANGO_CALLS_PIPELINE_ROOT=<MAIN_HOME>/.mango_local/mango_calls_two_processes
MANGO_CALLS_REMOTE_DROP_ROOT=/Users/Shared/mango_calls_drop_ro
MANGO_CALLS_REMOTE_INCOMING_ROOT=<MAIN_HOME>/.mango_local/mango_calls_remote_incoming
```

После создания файла подставить фактический HOME основного Mac, адрес M1 и
подтверждённый SHA, затем проверить отсутствие placeholders и разобрать env без
печати значений:

```bash
set -euo pipefail
MAIN_RECEIVER_ENV="$HOME/.mango_secrets/mango_calls_main_receiver.env"
M1_REMOTE_HOST='dmitriy@ИМЯ-ИЛИ-IP-M1'
EXPECTED_CODE_SHA='<ПОДТВЕРЖДЁННЫЙ_SHA>'
[[ "$M1_REMOTE_HOST" != *'<'* && "$EXPECTED_CODE_SHA" != *'<'* ]]
python3 - "$MAIN_RECEIVER_ENV" "$HOME" "$M1_REMOTE_HOST" "$EXPECTED_CODE_SHA" <<'PY'
import os, pathlib, sys
path = pathlib.Path(sys.argv[1])
text = path.read_text(encoding='utf-8')
text = text.replace('<MAIN_HOME>', sys.argv[2])
text = text.replace('dmitriy@M1_HOST', sys.argv[3])
text = text.replace('<тот же подтверждённый SHA>', sys.argv[4])
temporary = path.with_name(path.name + '.tmp')
temporary.write_text(text, encoding='utf-8')
os.chmod(temporary, 0o600)
os.replace(temporary, path)
PY
chmod 600 "$MAIN_RECEIVER_ENV"
! grep -q '<' "$MAIN_RECEIVER_ENV"
python3 scripts/mango_calls_env.py "$MAIN_RECEIVER_ENV"
```

Сначала только отрисовать plist:

```bash
set -euo pipefail
/usr/bin/python3 scripts/install_mango_calls_two_processes_service.py \
  --config ~/.mango_local/mango_calls_two_processes/config.json \
  --env-file ~/.mango_secrets/mango_calls_main_receiver.env \
  --process-b-only --process-b-interval-seconds 900 \
  --out-dir ~/.mango_local/mango_calls_two_processes/launchd-preview
```

В plist должен быть только `com.mango.calls-process-b` с командой
`process-b-pull`. Установка блокируется, пока локальный Process A загружен.
После проверки убрать старые задания в `MAIN_SNAP`. Установку B пока не выполнять:
первый контрольный цикл запускается вручную.

```bash
set -euo pipefail
MAIN_SNAP="$HOME/.mango_local/mango_calls_cutover/$(date -u +%Y%m%dT%H%M%SZ)_main"
mkdir -p "$MAIN_SNAP" && chmod 700 "$MAIN_SNAP"
printf '%s\n' "$MAIN_SNAP" > "$HOME/.mango_local/mango_calls_cutover/MAIN_SNAPSHOT_PATH"
chmod 600 "$HOME/.mango_local/mango_calls_cutover/MAIN_SNAPSHOT_PATH"
cp -p ~/Library/LaunchAgents/com.mango.calls-*.plist "$MAIN_SNAP/" 2>/dev/null || true
cp -p ~/.mango_local/mango_calls_two_processes/config.json "$MAIN_SNAP/"
PIPELINE_ROOT="$(/usr/bin/plutil -extract pipeline_root raw -o - \
  ~/.mango_local/mango_calls_two_processes/config.json)"
shasum -a 256 "$PIPELINE_ROOT/drop/"* > "$MAIN_SNAP/drop.sha256"
launchctl bootout "gui/$(id -u)/com.mango.calls-process-a" 2>/dev/null || true
launchctl bootout "gui/$(id -u)/com.mango.calls-two-processes" 2>/dev/null || true
mv ~/Library/LaunchAgents/com.mango.calls-process-a.plist "$MAIN_SNAP/" 2>/dev/null || true
mv ~/Library/LaunchAgents/com.mango.calls-two-processes.plist "$MAIN_SNAP/" 2>/dev/null || true
```

Каждый цикл основного Mac забирает manifest, затем SQLite, затем manifest ещё
раз. Принимаются только неизменившийся manifest, совпавшие SHA-256 и размер,
`quick_check=ok` и `integrity_check=ok`. DB заменяется первой, manifest последним;
предыдущее поколение хранится одним локальным rollback-hardlink. Только после
успешной приёмки выполняется Process B.

## Первый контрольный цикл

1. На M1 вручную запустить Process A и дождаться готовой ready-базы:
   ```bash
   set -euo pipefail
   /bin/zsh scripts/run_mango_calls_process.sh \
     ~/.mango_local/mango_calls_two_processes/config.json \
     ~/.mango_secrets/mango_calls_m1_worker.env process-a-worker
   ```
2. Проверить, что Process B на M1 не запущен.
3. На основном Mac вручную запустить получение и Process B:
   ```bash
   set -euo pipefail
   /bin/zsh scripts/run_mango_calls_process.sh \
     ~/.mango_local/mango_calls_two_processes/config.json \
     ~/.mango_secrets/mango_calls_main_receiver.env process-b-pull
   ```
4. Сверить SHA ready-базы на обоих компьютерах, зелёный отчёт Process B и зелёный
   закрытый суточный баланс из Фазы 0. Наличие файла без полного баланса не PASS.
5. Проверить XLSX, TXT и фактическую синхронизацию Яндекс Диска. Google-таблицу
   и отсутствие файла `ПРОВЕРКА — НЕ ИСПОЛЬЗОВАТЬ` проверять только если Google
   явно включён.
6. Проверить второй неизменный запуск: повторного скачивания итогов и дубля
   XLSX/TXT или включённой Google-таблицы быть не должно.

## Атомарный cutover после PASS ручного цикла

Нужно новое подтверждение владельца. `--install` записывает plist и сразу
выполняет `launchctl bootstrap`; отдельной безопасной установки без запуска нет.

На M1:

```bash
set -euo pipefail
/usr/bin/python3 scripts/install_mango_calls_two_processes_service.py \
  --config ~/.mango_local/mango_calls_two_processes/config.json \
  --env-file ~/.mango_secrets/mango_calls_m1_worker.env \
  --process-a-only --process-a-interval-seconds 1800 --install
```

На основном Mac:

```bash
set -euo pipefail
/usr/bin/python3 scripts/install_mango_calls_two_processes_service.py \
  --config ~/.mango_local/mango_calls_two_processes/config.json \
  --env-file ~/.mango_secrets/mango_calls_main_receiver.env \
  --process-b-only --process-b-interval-seconds 900 --install
```

После обеих команд проверить фактические PID, cwd, SHA, config/env, label и
lock. На M1 допустим только Process A, на основном Mac - только Process B.
Перезагрузка M1 должна вернуть ровно одну ожидаемую службу.

## Откат

Если первый контрольный цикл не прошёл, на M1 выключить новый Process A и
вернуть только те прежние задания, которые реально были в `M1_SNAP`:

```bash
set -euo pipefail
M1_SNAP="$(cat "$HOME/.mango_local/mango_calls_cutover/M1_SNAPSHOT_PATH")"
test -n "$M1_SNAP" && test -f "$M1_SNAP/config.json" && test -f "$M1_SNAP/active_labels.txt"
launchctl bootout "gui/$(id -u)/com.mango.calls-process-a" 2>/dev/null || true
cp -p "$M1_SNAP/config.json" ~/.mango_local/mango_calls_two_processes/config.json
while IFS= read -r label; do
  [[ "$label" == com.mango.calls-process-a || "$label" == com.mango.calls-process-b \
    || "$label" == com.mango.calls-two-processes ]]
  plist="$M1_SNAP/$label.plist"
  test -f "$plist"
  cp -p "$plist" ~/Library/LaunchAgents/
  launchctl bootstrap "gui/$(id -u)" ~/Library/LaunchAgents/"$label.plist"
done < "$M1_SNAP/active_labels.txt"
```

На основном Mac выключить раздельный Process B, проверить неизменность старого
источника и вернуть только те прежние plist, которые были реально загружены до
начала переноса. Новую локальную копию не удалять:

```bash
set -euo pipefail
SOURCE_SNAP="$(cat "$HOME/.mango_local/mango_calls_cutover/MAIN_SOURCE_SNAPSHOT_PATH")"
test -d "$SOURCE_SNAP" && test -f "$SOURCE_SNAP/SOURCE_CONFIG_PATH" \
  && test -f "$SOURCE_SNAP/drop.sha256" && test -f "$SOURCE_SNAP/active_labels.txt"
SOURCE_CONFIG="$(cat "$SOURCE_SNAP/SOURCE_CONFIG_PATH")"
test -f "$SOURCE_CONFIG" && test ! -L "$SOURCE_CONFIG"
OLD_PIPELINE="$(/usr/bin/plutil -extract pipeline_root raw -o - "$SOURCE_CONFIG")"
[[ "$OLD_PIPELINE" == /* && "$OLD_PIPELINE" != / ]]
launchctl bootout "gui/$(id -u)/com.mango.calls-process-b" 2>/dev/null || true
(cd "$OLD_PIPELINE/drop" && shasum -a 256 -c "$SOURCE_SNAP/drop.sha256")
while IFS= read -r label; do
  [[ "$label" == com.mango.calls-process-a || "$label" == com.mango.calls-process-b \
    || "$label" == com.mango.calls-two-processes ]]
  plist="$SOURCE_SNAP/$label.plist"
  test -f "$plist"
  cp -p "$plist" ~/Library/LaunchAgents/
  launchctl bootstrap "gui/$(id -u)" ~/Library/LaunchAgents/"$label.plist"
done < "$SOURCE_SNAP/active_labels.txt"
```

Reverse-copy базы с M1 запрещён без отдельного расследования. Старые данные не
удалять до семи последовательных зелёных суточных циклов и решения владельца.

Если принятая удалённая ready-база оказалась логически непригодной, сначала
выключить Process B, затем из чистой подтверждённой ревизии выполнить локальный
обмен текущего и предыдущего поколений:

```bash
set -euo pipefail
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
скорости, разделения ролей, отсутствия дублей, публикации Яндекс и, если явно
включено, Google, а также
доставки ready-базы в Timeline. Его точная карта приёмки находится в
`tasks/_inbox_codex/2026-07-31_TZ_m1_calls_stage10_pilot.md`.
