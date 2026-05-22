# Runbook для запуска на MacBook M1 Pro

## 1. Назначение

Этот документ описывает безопасную схему переноса внутренних Telegram-ботов и ядра ИИ-сотрудника на отдельный MacBook M1 Pro или внутренний сервер.

Это не инструкция для запуска прямо сейчас. Это проект runbook. Реальный перенос делать после 1-2 дней устойчивого пилота и отдельного подтверждения.

## 2. Базовые правила

Нельзя:

- хранить токены в git;
- писать в AMO/Tallanto/CRM без отдельного разрешения;
- запускать ASR/R+A;
- менять `stable_runtime`;
- отправлять клиентам ответы в обход матрицы автономности;
- держать Фотон и УНПК в одном неразделённом env без явных brand keys.

Можно:

- запускать Telegram polling;
- читать AMO/Tallanto через `api.fotonai.online` в read-only режиме;
- читать локальную базу знаний;
- писать локальные логи;
- писать локальную pilot DB;
- делать backup product DB;
- строить отчёты по пилоту.

## 3. Рекомендуемая структура папок

```text
/Users/shared/mango-ai-employee/
  app/
    Mango analyse/                # рабочая копия проекта
  product_data/
    db/
      ai_employee_pilot.sqlite
      channel_runtime.sqlite
    knowledge/
      active -> kb_release_...
    logs/
      telegram/
      service/
      errors/
    reports/
      pilot_daily/
      semantic_review/
    backups/
      daily/
  secrets/
    telegram_foton.env
    telegram_unpk.env
    crm_readonly.env
  run/
    pids/
    health/
```

Секреты не должны лежать внутри git-репозитория.

## 4. Env-файлы

Текущий локальный путь по runbook:

```text
/Users/dmitrijfabarisov/.codex/mango_telegram_pilot_bots.env
```

Для M1 Pro лучше разделить:

```text
secrets/telegram_foton.env
secrets/telegram_unpk.env
secrets/crm_readonly.env
```

Минимальные переменные:

```text
TELEGRAM_FOTON_BOT_TOKEN=...
TELEGRAM_UNPK_BOT_TOKEN=...
MANGO_TELEGRAM_CRM_READ_MODE=server
MANGO_CRM_SERVER_URL=https://api.fotonai.online
MANGO_CRM_SERVER_API_KEY=...
TELEGRAM_PILOT_KB_SNAPSHOT_PATH=...
TELEGRAM_PILOT_KILL_SWITCH=0
```

Правило: в отчёты можно писать только `token_present=true`, но не значение токена.

## 5. Запуск ботов

Проверка токенов:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/run_telegram_public_pilot_bots.py \
  --mode getme \
  --brand all
```

Короткий запуск:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/run_telegram_public_pilot_bots.py \
  --mode poll \
  --brand all \
  --duration-sec 60
```

Постоянный запуск:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/run_telegram_public_pilot_bots.py \
  --mode poll \
  --brand all
```

Для M1 Pro лучше оформить это через `launchd` или другой supervisor, но сначала достаточно ручного запуска и healthcheck.

## 6. Перезапуск

Безопасная последовательность:

1. Включить kill switch.
2. Дождаться остановки polling.
3. Проверить, что новых сообщений не обрабатывается.
4. Обновить код или KB snapshot.
5. Проверить `getme`.
6. Запустить на 60 секунд.
7. Проверить логи.
8. Запустить постоянно.

Не использовать `kill -9`, если процесс можно остановить штатно.

## 7. Проверка, что процесс жив

Минимально:

- процесс `run_telegram_public_pilot_bots.py` есть;
- лог обновлялся в последние 5 минут;
- нет повторяющихся ошибок;
- `getme` проходит;
- `.codex_local/telegram_pilot_bots/logs/` пополняется.

Будущий healthcheck должен показывать:

- bot username;
- active brand;
- KB release id;
- CRM read mode;
- last update received;
- last answer sent;
- last error;
- kill switch status;
- autonomous replies enabled/disabled.

## 8. Логи

Текущий путь:

```text
.codex_local/telegram_pilot_bots/logs/
```

На M1 Pro:

```text
product_data/logs/telegram/YYYY-MM-DD/*.jsonl
product_data/logs/service/YYYY-MM-DD/*.log
```

Log rotation:

- ежедневный файл;
- хранить подробные логи 30 дней;
- агрегированные отчёты хранить дольше;
- токены и raw payload не писать.

## 9. Backup

Ежедневно сохранять:

- SQLite pilot DB;
- channel DB;
- active KB snapshot;
- config без секретов;
- daily reports;
- feedback register.

Не обязательно копировать:

- временные raw logs старше retention;
- кэши модели;
- `.DS_Store`;
- `__pycache__`.

Backup должен проверяться через restore dry-run:

```text
backup exists -> sqlite integrity ok -> schema ok -> last report readable
```

## 10. Проверка AMO/Tallanto/API

Текущий режим для пилота:

```text
MANGO_TELEGRAM_CRM_READ_MODE=server
MANGO_CRM_SERVER_URL=https://api.fotonai.online
```

Проверять:

- API key присутствует локально;
- статус AMO active;
- Tallanto read-only отвечает;
- по тестовому телефону возвращается ожидаемый контекст;
- в логах нет live-write.

Нельзя:

- передавать прямой AMO token в бота;
- писать в AMO/Tallanto из Telegram runtime;
- раскрывать клиенту внутренние source ids.

## 11. Если Codex/LLM недоступен

Fail-closed поведение:

- не выдумывать ответ;
- не отправлять клиенту автономный ответ;
- дать безопасный fallback;
- передать менеджеру;
- записать `llm_unavailable`;
- не повторять запрос бесконечно.

Пример безопасного ответа:

```text
Сейчас передам вопрос менеджеру, чтобы не дать неточную информацию.
```

Но для зелёных фактов можно использовать заранее проверенные короткие ответы только если они не требуют LLM.

## 12. Как временно отключить автоответы

Должны быть отдельные выключатели:

```text
TELEGRAM_PILOT_KILL_SWITCH=1
TELEGRAM_PILOT_AUTONOMY_ENABLED=0
TELEGRAM_PILOT_CLIENT_SEND_ENABLED=0
```

При любом сомнении:

- client send off;
- manager draft only;
- CRM/Tallanto read-only remains allowed;
- logs continue.

## 13. Как переключить модель или уровень рассуждения

Переключение модели должно быть конфигом, а не правкой кода.

Нужно логировать:

- model name;
- reasoning effort;
- prompt version;
- KB release;
- route;
- latency.

Если новая модель даёт больше P0/P1 или шаблонных ответов, откатить модель и зафиксировать incident.

## 14. Как вернуть предыдущую версию базы знаний

Нужно хранить:

- active KB path;
- previous KB path;
- checksum;
- semantic status;
- release notes.

Откат:

1. Включить autonomy off или kill switch.
2. Поменять `TELEGRAM_PILOT_KB_SNAPSHOT_PATH` на предыдущую проверенную версию.
3. Запустить короткий smoke.
4. Проверить Фотон/УНПК brand separation.
5. Вернуть polling.

Нельзя откатываться на версию без `semantic_pass`, если нет отдельного решения.

## 15. Когда переносить на M1 Pro

Не раньше, чем:

- два бота стабильно отвечают на текущем MacBook;
- есть daily feedback report;
- нет P0 в живых проверках;
- известен active KB snapshot;
- понятен процесс остановки;
- есть хотя бы базовый backup.

Перенос на M1 Pro должен быть операционным шагом, а не архитектурным экспериментом.
