# Runbook Mango

Обновлено: 2026-07-19.

Этот файл содержит текущие безопасные команды. Исторические команды и решения
остаются в Git и `docs/DECISIONS_LOG.md`, но не считаются рабочей инструкцией.

## Перед любой работой

```bash
git status --short --branch
python3 scripts/project_now.py
sed -n '1,220p' docs/PROJECT_NOW.md
sed -n '1,220p' docs/DECISIONS_LOG.md
```

Для крупного ТЗ после штатного переноса в `tasks/_running`:

```bash
python3 scripts/skills/tz_lint.py tasks/_running/<TZ.md>
python3 scripts/preflight.py --tz tasks/_running/<TZ.md>
```

Не начинать изменяющий блок в чужом или грязном worktree. Не использовать
`git add -A`.

## Источник факта о live

Основная папка:

```text
/Users/dmitrijfabarisov/Projects/Mango analyse
```

Wappi, calls A/B и customer-timeline nightly настроены на эту папку. Перед
деплоем или изменением флагов обязателен жёсткий гейт:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  python3 scripts/skills/live_truth.py --no-write
```

`PASS` допустим только когда живой PID связан с launchd, startup manifest и
heartbeat, а загруженный SHA совпадает с текущим HEAD. `WARN` означает drift
или неполную аттестацию; `NO_PROCESS` означает, что ожидаемый процесс не найден.
Оба статуса блокируют live-действие.

Дополнительно проверять фактические LaunchAgents:

```bash
launchctl print gui/$(id -u)/com.mango.wappi-draft-loop
launchctl print gui/$(id -u)/com.mango.calls-process-a
launchctl print gui/$(id -u)/com.mango.calls-process-b
launchctl print gui/$(id -u)/com.mango.customer-timeline-nightly
```

Путь в plist сам по себе не доказывает, какой код загружен в уже работающий
Python. Нужны PID, cwd, env, startup manifest и heartbeat.

## Wappi draft-loop

Read-only проверка:

```bash
cat /Users/dmitrijfabarisov/.mango_local/draft_loop/heartbeat.json
cat /Users/dmitrijfabarisov/.mango_local/draft_loop/phase1b_startup_manifest.json
tail -100 /Users/dmitrijfabarisov/.mango_local/draft_loop/launchd.stderr.log
```

Штатный installer с проверкой единственного процесса, `live_truth` и
автоматическим возвратом предыдущего plist:

```bash
./scripts/start_wappi_draft_loop_launchd.sh
```

Команда меняет live-службу. Запускать её только по отдельному подтверждённому
ТЗ. Не запускать `run_amo_wappi_draft_loop.py --live-write` вручную вместо
installer.

Wappi создаёт только менеджерскую заметку-черновик в AMO. Автоотправки клиенту
нет.

## Звонки: процессы A/B

Текущая схема:

- `process-a` запускается каждые 1800 секунд;
- `process-b` запускается по требованию и не имеет интервала;
- оба используют `scripts/run_mango_calls_process.sh` из основной папки.

Read-only проверка:

```bash
launchctl print gui/$(id -u)/com.mango.calls-process-a
launchctl print gui/$(id -u)/com.mango.calls-process-b
tail -100 product_data/mango_calls_two_processes/logs/process-a.stderr.log
tail -100 product_data/mango_calls_two_processes/logs/process-b.stderr.log
```

Штатная переустановка, только по отдельному подтверждённому ТЗ:

```bash
python3 scripts/install_mango_calls_two_processes_service.py \
  --config /Users/dmitrijfabarisov/.mango_local/mango_calls_two_processes/config.json \
  --env-file /Users/dmitrijfabarisov/.mango_secrets/mango_office.env \
  --process-a-interval-seconds 1800 \
  --install
```

Старый label `com.mango.calls-two-processes` не является действующим
конвейером A/B.

## Customer Timeline nightly

Текущий запуск: ежедневно в 03:30 из основной папки.

Единственный почтовый архив хранится вне репозитория:

```text
/Users/dmitrijfabarisov/Mango_Data/_external_handoffs/mail_archive_canonical_20260711
```

Все рабочие почтовые команды используют корень `MANGO_MAIL_DATA_ROOT`; его
штатное значение — `/Users/dmitrijfabarisov/Mango_Data`. Старой копии архива
внутри репозитория быть не должно.

Read-only проверка службы:

```bash
launchctl print gui/$(id -u)/com.mango.customer-timeline-nightly
tail -100 /Users/dmitrijfabarisov/.mango_local/customer_timeline_nightly/.codex_local/staging/nightly_service/launchd.stderr.log
```

Проверка текущей SQLite без записи:

```bash
sqlite3 \
  'product_data/customer_timeline/customer_timeline_prod_20260621/customer_timeline.sqlite' \
  'PRAGMA query_only=ON; PRAGMA quick_check;'
```

Штатная установка, только по отдельному подтверждённому ТЗ:

```bash
bash scripts/install_customer_timeline_nightly_service.sh \
  --plist deploy/customer_timeline_nightly/com.mango.customer-timeline-nightly.plist.template \
  --code-root "$PWD" \
  --nightly-home /Users/dmitrijfabarisov/.mango_local/customer_timeline_nightly \
  --apply
```

Installer не снимает уже загруженный label. Если служба существует, не
повторять команду поверх неё: переустановка оформляется отдельным cutover-ТЗ с
проверкой текущего plist и возвратом при ошибке.

## База знаний

Текущий snapshot:

```text
product_data/knowledge_base/kb_release_20260612_v6_7_staging_r4_1/
  kb_release_v3_snapshot.json
```

Не подменять его старым v6.3 из исторических документов. Бот может называть
цену, дату, расписание, адрес и условия только из подтверждённых client-safe
фактов нужного бренда и области.

## Тесты

Безопасный сбор:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  python3 -m pytest --collect-only -q
```

Точечный запуск:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  python3 -m pytest -q <tests>
```

Импорт ядра:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -c \
  "from mango_mvp.channels.subscription_llm_parts import SubscriptionLlmDraftProvider"
```

Для клиентских ответов, базы знаний, CRM/AMO/Tallanto-текстов и коммерческих
фактов зелёные тесты означают только `formal_pass`. До вывода о готовности нужен
отдельный `semantic_pass` по `docs/SEMANTIC_REVIEW_RULES.md`.

## Что запрещено без отдельного подтверждения

- ASR и Resolve+Analyze по реальным данным;
- live-write в AMO/CRM/Tallanto;
- отправка сообщений клиентам;
- изменение `stable_runtime` и боевой customer timeline;
- тяжёлые batch/start/run-ui скрипты;
- удаление или перенос runtime, worktree, веток, тегов и баз.

## AMO snapshot и rollback

Любой новый live-write блок обязан заранее иметь snapshot, readback и rollback.
Rollback сначала запускается только в dry-run. Реальный rollback требует
отдельного подтверждения владельца.

Текущий Wappi-контур пишет только черновик-заметку. Это всё равно live-write в
AMO и не должно запускаться вручную вне штатной службы.

## Audit pack и коммит

После значимого блока создать один каталог:

```text
audits/_inbox/<block>_<timestamp>/
```

Минимум: `implementation_notes.md`, `changed_files.txt`, `test_output.txt`,
`risk_review.md`, `backward_compatibility.md`; для клиентского содержания также
`semantic_review.md`.

Перед коммитом:

1. проверить `git diff` и `git diff --check`;
2. запустить заявленные тесты;
3. убедиться, что нет runtime, ПДн, секретов и чужих изменений;
4. добавить только явные файлы своего блока;
5. проверить staged diff перед commit/push.

## Навыки проекта

- `scripts/skills/tz_lint.py` — advisory-проверка ТЗ;
- `scripts/skills/inventory_before_build.py` — поиск существующей реализации;
- `scripts/skills/fail_raw_export.py` — обязательное сырое доказательство FAIL;
- `scripts/skills/wappi_draft_loop_replay.py` — read-only replay перед изменением
  Wappi-петли;
- `scripts/skills/live_truth.py` — жёсткий гейт перед live-действиями;
- `mango-graphify` — локальная read-only карта, вывод проверяется в исходниках.
