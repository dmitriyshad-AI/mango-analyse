# Handoff: AI Office server access, TZ-17, AMO draft-loop

Дата: 2026-06-12.

Назначение: кратко зафиксировать, что Codex сделал после последнего промта Claude по ТЗ-17, какие возможности теперь есть по серверу AI Office, что уже проверено, и что передать Claude для сверки.

Секреты и приватные ключи в этот файл не внесены.

## 1. Что было сделано по ТЗ-17

### Черновиковый контур восстановлен

Сняты оба стоп-файла:

- `~/.mango_secrets/STOP_DRAFT_LOOP`
- `~/.mango_local/draft_loop/STOP_DRAFT_LOOP`

Перезапущены постоянные процессы:

- `screen`: `mango_draft_loop`
- `screen`: `mango_draft_loop_watchdog`

Проверка пульса:

- `status=ok`
- `stop_active=false`
- `auth_error_count=0`
- live-loop работает

Контроль задних черновиков:

- `retro_violations=0`
- старые сообщения, которые были до `not_before_ts`, не породили новые черновики

### Auto-resolver: файл кандидатов на сверку

Постоянный файл вне репозитория:

```text
~/.mango_local/draft_loop_inventory/auto_pairs_for_review_2026-06-12.json
```

Содержимое:

- просмотрено `5012` Wappi-чатов;
- `4956` приватных диалогов;
- `1270` диалогов с последним входящим текстом;
- найдено `76` кандидатов `matched`;
- файл содержит ПДн и поэтому не добавлен в git.

Метод:

- Wappi full chat listing;
- локальный AMO snapshot `product_data/customer_profiles/tz14_amo_step1_full_20260612/amo_step1_snapshot.sqlite`;
- read-only;
- без записи в AMO;
- без создания auto-pairs;
- без вызова модели.

Примечание: live AMO-MCP dry-run без лимита был запущен, но оказался слишком долгим: за ~16 минут дошёл до 213 строк и 22 кандидатов. Он был остановлен, чтобы не держать долгий foreground-процесс. Полная инвентаризация сделана через локальный snapshot.

### Stoplist общих семейных телефонов для Max

Файл вне репозитория:

```text
~/.mango_secrets/shared_phones_stoplist.json
```

Источник:

```text
product_data/customer_profiles/tz14_amo_step1_full_20260612/common_phone_review.csv
```

Результат:

- `112` валидных телефонов;
- мусорное значение `+7` отфильтровано;
- подключено к Max auto-resolver через путь `shared_phones_stoplist.json`;
- сохранён fallback на старое имя `shared_phone_stoplist.json`.

### Инструкция по server-side allowlist

Создан файл:

```text
D1_audit_backlog/INSTRUKCIYA_server_allowlist_update.md
```

В нём описано:

- где на сервере AI Office лежит allowlist;
- почему есть два уровня допуска:
  - `AMO_NOTE_HARD_ALLOWED_LEAD_IDS`
  - `CRM_AMO_NOTE_ALLOWED_LEAD_IDS`
- как делать backup;
- как перезапускать только API;
- как проверять `401`, `403`, `200`;
- что не трогать;
- как выдать безопасный deploy-доступ.

### Коммит Mango

Коммит:

```text
ef9747c0 Restore draft loop and document allowlist path
```

Вошли файлы:

- `src/mango_mvp/integrations/draft_loop.py`
- `scripts/run_amo_wappi_draft_loop.py`
- `tests/test_draft_loop.py`
- `tests/test_run_amo_wappi_draft_loop.py`
- `D1_audit_backlog/INSTRUKCIYA_server_allowlist_update.md`
- `tasks/_done/2026-06-12_TZ17_restore_loop_report.md`

Тесты:

```text
3067 passed, 2 skipped, 1 warning
```

## 2. Что сделано по AI Office allowlist

В локальной чистой копии AI Office был подготовлен коммит:

```text
d0f2f5d0 Allow AMO draft notes for pilot leads
```

Он добавляет в allowlist сделки:

```text
49832125, 47854947, 49325789, 49762441
```

Проверено локальными unit-тестами AI Office:

```text
7 passed
```

После этого другой серверный диалог внёс те же изменения на сервер `/opt/ai-office`.

На сервере сейчас подтверждено:

- `AMO_NOTE_HARD_ALLOWED_LEAD_IDS = {49832125, 47854947, 49325789, 49762441}`
- default `CRM_AMO_NOTE_ALLOWED_LEAD_IDS = 49832125,47854947,49325789,49762441`
- API-контейнер `ai-office-api-1` работает
- проверка без ключа даёт `401`
- проверка чужой сделки с ключом даёт `403`
- live-write `200` специально не выполнялся, чтобы не создавать тестовое примечание в AMO без отдельной команды

## 3. Новый серверный доступ Codex

Проверен root SSH-доступ к серверу AI Office.

Параметры доступа зафиксированы вне git в локальном файле:

```text
~/.mango_secrets/server_access_packs/ai_office_root_access_20260612.md
```

Проверено:

- `whoami -> root`
- `hostname -> nl-vmv2-mini`
- `/opt/ai-office` доступен
- `docker compose ps api` показывает `ai-office-api-1 Up`

Важно:

- на сервере рабочее дерево `/opt/ai-office` грязное;
- в дереве есть live-изменения AI Office;
- `git reset`, `git clean`, `git checkout` запрещены без отдельной явной команды Дмитрия;
- `.env` и секреты не печатать;
- live-write в AMO не делать без отдельного подтверждения.

## 4. Что теперь может делать Codex

Codex теперь может самостоятельно:

- подключаться к серверу AI Office;
- читать `/opt/ai-office`;
- смотреть `git status` и текущий diff;
- смотреть состояние API-контейнера;
- смотреть логи API;
- делать timestamped backup файлов перед правками;
- точечно править серверные файлы при явном ТЗ;
- перезапускать только API через `docker compose up -d --build api`;
- проверять публичные endpoint-ответы `401/403`;
- при отдельном разрешении Дмитрия выполнять live-write проверку `200`, понимая, что она создаёт реальное примечание в AMO.

Codex не должен без отдельной команды:

- отправлять сообщения клиентам;
- делать live-write в AMO;
- читать или печатать `.env`;
- делать destructive git-команды;
- деплоить чужие dirty-изменения вслепую.

## 5. Текущие ограничения и риски

1. Server AI Office tree dirty.

   Это не ошибка, а текущий live-state. Любой серверный deploy должен быть точечным, с backup, без `git pull/reset`.

2. Full live AMO-MCP auto-resolve слишком медленный.

   Полный список кандидатов на сверку получен через Wappi + локальный AMO snapshot. Для постоянного production auto-resolver лучше делать отдельный оптимизированный режим с кэшем/батчингом.

3. Auto-pairs не включены автоматически.

   Найденные 76 кандидатов ждут сверки архитектора/Дмитрия. В live-loop auto-resolver сейчас выключен, ручные пары работают отдельно.

4. Max stoplist собран по свежему TZ14 common-phone review.

   ТЗ упоминало `~367`, но локально свежий источник дал 112 валидных телефонов. Это зафиксировано как расхождение источников.

## 6. Промт для Claude

```text
Claude, изучи результаты работы Codex по ТЗ-17 и серверному доступу AI Office.

Файлы и артефакты:

1. Отчёт ТЗ-17:
   tasks/_done/2026-06-12_TZ17_restore_loop_report.md

2. Инструкция по server-side allowlist:
   D1_audit_backlog/INSTRUKCIYA_server_allowlist_update.md

3. Handoff-файл:
   D1_audit_backlog/HANDOFF_ai_office_server_access_tz17_2026-06-12.md

4. Внешний файл кандидатов auto-resolver, содержит ПДн, не в git:
   ~/.mango_local/draft_loop_inventory/auto_pairs_for_review_2026-06-12.json

5. Внешний stoplist общих телефонов Max, не в git:
   ~/.mango_secrets/shared_phones_stoplist.json

6. Локальная памятка root-доступа AI Office, не в git:
   ~/.mango_secrets/server_access_packs/ai_office_root_access_20260612.md

Что нужно проверить:

- ТЗ-17 действительно закрыто по смыслу:
  - стоп-файлы сняты;
  - loop/watchdog подняты;
  - heartbeat ok;
  - задних черновиков нет;
  - auto-pair candidates вынесены во внешний файл;
  - Max stoplist подключён;
  - server allowlist procedure описана.

- Проверь, корректен ли выбор stoplist-источника:
  Codex взял fresh TZ14 `common_phone_review.csv`, 112 валидных номеров.
  ТЗ говорило `identity_links ~367`, но локальные источники не подтвердили это число.
  Нужен вердикт: достаточно 112 или нужен другой источник.

- Проверь файл auto_pairs_for_review_2026-06-12.json:
  - 5012 Wappi-чата просмотрено;
  - 1270 latest inbound text;
  - 76 matched candidates;
  - нет ли очевидных ложных привязок;
  - какие пары можно предложить Дмитрию для server allowlist/manual enable.

- Проверь server-side allowlist:
  На сервере добавлены `49832125,47854947,49325789,49762441`.
  401/403 проверены, live-write 200 не делался.
  Нужно ли делать live-write проверку на одной разрешённой сделке или ждать реального входящего.

- Проверь, достаточно ли нового root-доступа для будущих серверных задач:
  Codex теперь может читать/править/backup/restart API на `/opt/ai-office`, но дерево dirty.
  Нужно сформулировать безопасный процесс серверных изменений без `git reset/clean/checkout`.

Отдельно:
- Не проси Codex печатать секреты или `.env`.
- Не проси live-write в AMO без явной команды Дмитрия.
- Если предлагаешь включать auto-pairs, дай список candidate lead_id/chat/profile и причину доверия, но ПДн держи вне git.
```
