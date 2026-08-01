# Промпт для новой задачи Codex на M1

```text
Ты работаешь на новом Mac M1 и готовишь безопасный запуск полного контура
обработки звонков Mango. Репозиторий должен находиться по пути:
/Users/dmitrijfabarisov/Projects/Mango analyse

Сначала прочитай целиком:
1. AGENTS.md
2. README.md
3. ARCHITECTURE.md
4. docs/PROJECT_NOW.md; если старше суток, запусти python3 scripts/project_now.py
5. docs/RUNBOOK.md
6. docs/DECISIONS_LOG.md
7. docs/m1_calls_handoff_20260801/README.md
8. docs/M1_MANGO_CALLS_SPLIT_CUTOVER_RUNBOOK.md
9. tasks/_inbox_codex/2026-07-31_TZ_m1_calls_stage10_pilot.md

Цель текущей задачи: установить зависимости, восстановить пользовательские
skills/plugins/MCP, проверить доступы и подготовить M1 к пилоту. НЕ запускать
ASR, Resolve, Analyze и службы без отдельной команды Дмитрия. НЕ переносить
SQLite, аудио и секреты через Git или Яндекс Диск.

Порядок:
1. Выполни git status --short --branch и проверь, что ветка main чистая и
   совпадает с origin/main. Покажи SHA. Все дальнейшие действия выполняй только
   из этого единого SHA. Он обязан совпасть с файлом
   `M1 Handoff 20260801/CANONICAL_GIT_SHA.txt` на Яндекс Диске; иначе остановись.
2. Запусти scripts/bootstrap_m1_mango_calls.sh plan.
3. Если Homebrew отсутствует, дай Дмитрию официальную команду с brew.sh и
   остановись. Иначе с явной переменной подтверждения выполни режим install.
4. Проверь codex login status. Если входа нет, попроси Дмитрия выполнить
   интерактивный codex login; не копируй auth.json через Git/чат.
5. Установи локальные skills из архива на Яндекс Диске, затем по
   MCP_RESTORE_CHECKLIST.md и codex_profile_manifest_20260801.json восстанови
   доступные plugins. GitHub, Google
   Drive и Todoist авторизуй заново через интерфейс; токены не копируй.
6. Проверь наличие файлов 0600:
   ~/.mango_secrets/mango_calls_m1_worker.env
   ~/.mango_secrets/mango_office.env
   ~/.mango_secrets/tallanto_readonly.env
   Не печатай их значения. Если файлов нет, дай Дмитрию точную команду scp из
   README и остановись.
7. Проверь наличие `~/.mango_local/tallanto/Contacts_current.csv` с режимом
   0600. Если файла нет, используй только прямой scp из README; через Git,
   Яндекс Диск или audit pack его не передавай.
8. Создай config.json из config.m1.example.json, подставив реальный HOME и путь
   к установленному Python. Создай пустой pipeline_root с режимом 0700, но не
   переноси в него данные без отдельной команды. Ничего не клади в stable_runtime.
9. Выполни scripts/bootstrap_m1_mango_calls.sh check и сообщи только поля
   true/false. Это локальная проверка наличия, а не доказательство доступа к API.
10. Отрисуй launchd plist только через --out-dir. Не используй --install.
11. Выполни безопасные точечные тесты из ТЗ этапа 10. Не запускай полный
    Process A и не делай сетевой Mango batch.
12. После отдельного подтверждения Дмитрия выполни минимальные read-only проверки:
    получение справочника пользователей Mango без скачивания звонков, чтение
    одной записи Tallanto и чтение метаданных закрытой Google-папки. Не делай
    запись и не печатай ответы с персональными данными; в отчёт внеси только
    success/fail, HTTP-класс и время.
13. Подготовь audit pack: версии, SHA, наличие модулей, локальных входов,
    результаты отдельных read-only проверок, свободное
    место, rendered plist, тесты, риски. ПДн и секреты исключи.
14. Только если все обязательные локальные и read-only проверки зелёные,
    формулируй статус как «M1 подготовлен к ручному cutover». При любом false
    выдай `BLOCKED` и конкретный следующий шаг. Никогда не пиши «работает в бою».
    Для cutover дождись отдельного подтверждения Дмитрия.

В финале выведи пять отдельных статусов: formal_pass, semantic_pass,
business_pass, data_pass и runtime_pass. До реального пилота runtime_pass=false,
pilot_ready=false и production_ready=false.

Обязательные ограничения:
- один Whisper worker и один GigaAM worker;
- Resolve/Analyze используют изолированный Codex без plugins/MCP;
- Google service account только в ~/.mango_secrets с режимом 0600;
- Process A не может одновременно работать на старом Mac и M1;
- Process B и production Timeline не запускаются на M1;
- никакой записи в AMO/CRM/Tallanto;
- никакого удаления старого runtime.
```
