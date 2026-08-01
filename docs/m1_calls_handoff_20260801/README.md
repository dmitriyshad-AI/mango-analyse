# Передача контура звонков на M1

Дата пакета: 2026-08-01.

Этот пакет готовит M1 к работе, но сам не переключает службу и не запускает
ASR, Resolve или Analyze. Переключение выполняется только по
`docs/M1_MANGO_CALLS_SPLIT_CUTOVER_RUNBOOK.md` после отдельного подтверждения
владельца.

## Что передаётся через Git

- код инкрементального получения звонков Mango;
- один последовательный контур из четырёх одиночных стадий: Whisper, GigaAM,
  Resolve и Analyze;
- строгая проверка ролей и порядка реплик;
- код создания и проверки запечатанного ready-снимка SQLite и его безопасной
  передачи на основной Mac, но не сама база;
- код создания суточных XLSX/TXT и публикации в закрытую Google-папку, но не
  сами отчёты;
- установщик пакетов `scripts/bootstrap_m1_mango_calls.sh`;
- runbook, тесты и стартовый промпт для Codex.

Через Git не передаются аудио, SQLite, расшифровки, ключи, токены, Codex login,
файлы Google service account и другие персональные данные.

## 1. Установка кода и пакетов на M1

1. Установить Xcode Command Line Tools и Homebrew. Homebrew ставить только с
   официального `https://brew.sh`, не из случайной копии команды.
2. Создать SSH-ключ на M1 и добавить публичную часть в GitHub либо выполнить
   `gh auth login`.
3. Клонировать репозиторий в тот же абсолютный путь:

   ```bash
   mkdir -p "$HOME/Projects"
   git clone git@github.com:dmitriyshad-AI/mango-analyse.git \
     "$HOME/Projects/Mango analyse"
   cd "$HOME/Projects/Mango analyse"
   git switch main
   git pull --ff-only
   EXPECTED_SHA="$(cat "$HOME/Yandex.Disk.localized/Mango Calls Resolve/M1 Handoff 20260801/CANONICAL_GIT_SHA.txt")"
   test "$(git rev-parse HEAD)" = "$EXPECTED_SHA"
   ```

4. Посмотреть план, затем установить пакеты. Скрипт ничего не запускает:

   ```bash
   scripts/bootstrap_m1_mango_calls.sh plan
   CONFIRM_M1_PACKAGE_INSTALL=INSTALL_M1_MANGO_CALLS_PACKAGES \
     scripts/bootstrap_m1_mango_calls.sh install
   ```

5. Войти в Codex CLI на M1 под тем же владельцем подписки:

   ```bash
   codex login
   codex login status
   ```

Resolve и Analyze используют отдельный изолированный профиль Codex без skills,
plugins и MCP. Это защита от скрытой зависимости результата от интерфейса
разработчика. Пользовательский профиль Codex переносится отдельно только для
разработки и аудита.

## 2. Доступы

Обязательны для Process A:

- `MANGO_OFFICE_API_KEY` и `MANGO_OFFICE_API_SALT`;
- подписочная авторизация Codex CLI;
- точный SHA чистой версии кода в `MANGO_CALLS_EXPECTED_CODE_SHA`.

Для полного внутреннего отчёта нужны:

- read-only доступ Tallanto;
- локальная актуальная CSV-выгрузка контактов Tallanto;
- установленный клиент Яндекс Диска и закрытый каталог
  `~/Yandex.Disk.localized/Mango Calls Resolve`.

Для Google нужны дополнительно. На этапе установки они могут ещё отсутствовать,
но до первого дня семисуточного пилота обязательны:

- отдельный service account JSON вне Git и Яндекс Диска;
- режим файла `0600` в `~/.mango_secrets/`;
- идентификатор закрытой папки Google Drive, куда service account имеет право
  добавлять файлы.

На основном Mac Mango и Tallanto уже настроены. Безопасного Google-файла для
этого контура на 2026-08-01 нет: найденный файл на Яндекс Диске использовать
нельзя. Google остаётся красным до отдельного создания или защищённого переноса
service account.

### Защищённый перенос Mango и Tallanto

После включения SSH на M1 выполнить на основном Mac, подставив реальный адрес:

```bash
M1_HOST='dmitrijfabarisov@ИМЯ-ИЛИ-IP-M1'
ssh "$M1_HOST" 'umask 077; mkdir -p ~/.mango_secrets; chmod 700 ~/.mango_secrets'
scp ~/.mango_secrets/mango_office.env "$M1_HOST":~/.mango_secrets/mango_calls_m1_worker.env
scp ~/.mango_secrets/mango_office.env "$M1_HOST":~/.mango_secrets/mango_office.env
scp ~/.mango_secrets/tallanto_readonly.env "$M1_HOST":~/.mango_secrets/tallanto_readonly.env
ssh "$M1_HOST" 'chmod 600 ~/.mango_secrets/*.env'
```

Команды не печатают значения. Не пересылать эти файлы через Git, чат, почту,
Яндекс Диск или audit pack.

В `mango_calls_m1_worker.env` на M1 вручную добавить строки из
`mango_calls_m1_worker.env.example`. `MANGO_CALLS_EXPECTED_CODE_SHA` должен
равняться значению из `CANONICAL_GIT_SHA.txt` и `git rev-parse HEAD`, а рабочая
папка должна быть чистой. Значения с пробелами оставлять в двойных кавычках.
Env-файл читает безопасный разборщик: команды, подстановки shell, дубли ключей и
некавыченные пробелы блокируются и не исполняются.

CSV Tallanto содержит персональные данные, поэтому её тоже нельзя передавать
через Git или audit pack. Передать напрямую с основного Mac:

```bash
ssh "$M1_HOST" 'umask 077; mkdir -p ~/.mango_local/tallanto; chmod 700 ~/.mango_local/tallanto'
scp -p '/Users/dmitrijfabarisov/Projects/Mango analyse/_external_handoffs/tallanto_contacts_export_2026-06-20/converted/Contacts 20.06.2026.csv' \
  "$M1_HOST":~/.mango_local/tallanto/Contacts_current.csv
ssh "$M1_HOST" 'chmod 600 ~/.mango_local/tallanto/Contacts_current.csv'
```

Это исходный снимок для первичного запуска. После запуска M1 обновление имён
дополняется read-only запросами Tallanto; периодическую замену CSV нужно включить
в отдельный регламент свежести. Поле `MANGO_CALLS_TALLANTO_SNAPSHOT_AS_OF`
фиксирует дату содержимого явно; дата изменения файла больше не является
источником правды для нового M1-контура.

## 3. Конфигурация

Создать `~/.mango_local/mango_calls_two_processes/config.json` по
`config.m1.example.json`, заменив `<HOME>` на фактический домашний каталог.
Файл должен иметь режим `0600` или более строгий, env - ровно `0600`.
Создать пустой `pipeline_root` из config с режимом `0700`; данные появятся там
только при отдельно подтверждённом первичном переносе.

Локальная проверка M1 без запуска службы:

```bash
cd "$HOME/Projects/Mango analyse"
MANGO_CALLS_CONFIG="$HOME/.mango_local/mango_calls_two_processes/config.json" \
MANGO_CALLS_ENV_FILE="$HOME/.mango_secrets/mango_calls_m1_worker.env" \
scripts/bootstrap_m1_mango_calls.sh check
```

Скрипт проверяет наличие и права локальных файлов, пакетов, каталога и SHA, но
намеренно не обращается к API. Поле `network_access_verified` всегда `false`.
До пилота все локальные поля, включая Google, должны стать `true`, после чего
отдельно выполняются минимальные read-only проверки Mango, Tallanto и метаданных
закрытой Google-папки. Не называйте наличие ключа подтверждённым доступом.

## 4. Skills, plugins и MCP Codex

Локальные skills передаются отдельным архивом
`Codex M1 Profile 20260801/codex_skills_clean_20260801.tar.gz` на Яндекс Диске.
В нём
нет `auth.json`, токенов или `config.toml`. На M1:

```bash
mkdir -p ~/.codex/skills
cd "$HOME/Yandex.Disk.localized/Mango Calls Resolve/Codex M1 Profile 20260801"
shasum -a 256 -c codex_profile_checksums_20260801.txt
tar -xzf codex_skills_clean_20260801.tar.gz \
  -C ~/.codex/skills
```

Plugins и MCP не копируются как токены. Установить Codex Desktop/ChatGPT на M1,
войти в тот же аккаунт, затем по `MCP_RESTORE_CHECKLIST.md` и
`codex_profile_manifest_20260801.json` включить те же доступные plugins. GitHub,
Google Drive и Todoist нужно заново авторизовать в интерфейсе. Наличие локального
кэша plugin не означает, что он уже установлен или авторизован. После этого
проверить:

```bash
codex mcp list
```

Ожидаются GitHub, встроенный `node_repl`, доступные через Desktop инструменты
браузера и компьютера; Todoist является connector, а не локальным секретом.
MCP не передаются в изолированный профиль Resolve/Analyze.

## 5. Данные и службы

Первичный перенос working DB и аудио выполняется напрямую по SSH/rsync по
разделу «Первичный перенос данных» в
`docs/M1_MANGO_CALLS_SPLIT_CUTOVER_RUNBOOK.md`. Git и Яндекс Диск для него
запрещены. После копирования обязательны `0700` для корня и рабочих каталогов,
`0600` для SQLite/JSON и обе проверки SQLite. До cutover текущий Process A
остаётся на основном Mac.

Установку launchd сначала только отрисовать. Команда с `--install`, остановка
старой службы и запуск первого M1-цикла требуют отдельного подтверждения
владельца. После переключения обязательны семь последовательных зелёных суток
по ТЗ `tasks/_inbox_codex/2026-07-31_TZ_m1_calls_stage10_pilot.md`.

## 6. Что проверить перед первым запуском

1. Одинаковый чистый SHA кода на двух Mac.
2. Не менее 40 ГиБ свободного места на M1.
3. Ровно один Whisper и один GigaAM worker, не несколько копий.
4. Mango/Tallanto доступны только на чтение там, где это предусмотрено.
5. Google-папка закрыта; service account не лежит в облачном диске.
6. Process B не запущен на M1, Process A не запущен одновременно на двух Mac.
7. Ready DB и manifest проходят SHA, размер, `quick_check` и `integrity_check`.
8. РОП получает полный диалог без обрезки, а сомнительные роли остаются на
   листе проблем.
9. `bootstrap ... check` зелёный и отдельно подтверждены read-only запросы к
   Mango, Tallanto и метаданным закрытой Google-папки.
