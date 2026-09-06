> TAKE 2026-09-06 19:12 | ветка codex/customer-timeline-source-gates-20260902 | codex

Ветка: codex/customer-timeline-source-gates-20260902
Зоны: scripts/run_customer_timeline_codex_task.py, tests/test_customer_timeline_codex_task.py, docs/DECISIONS_LOG.md, tasks/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_customer_timeline_codex_task.py tests/test_customer_timeline_nightly_service.py tests/test_adr003_regex_understanding_moratorium.py
Семантический-аудит: да
Feature-ID: feature.customer_timeline.nightly_wrapper_quality
Problem-ID: problem.customer_timeline.wrapper_nested_json_status
Изменение: fix
Ключевые-символы: parse_last_json,status_from_payload
Ключевые-слова: warning prefix nested JSON,nightly wrapper false stop,mail chain summary parser

# Ночной отчёт: читать внешний результат, а не вложенную диагностику

## Доказательство

HEAD332d6a8b. Run20260906T153346Z действительно overall_status=ok и
data_quality_status=pass,1955.27s,14/14ok,WAL[0,0,0]/0. Сохранённый service
report в audits/_inbox/customer_timeline_amo_window_entry_20260906/
first_service_strict_pass_153346.json. Wrapper rc1/stopped/data_quality_status_missing
при rc дочернего процесса0. Перед JSON есть urllib3 warning. parse_last_json
после неудачного json.loads перебирает КАЖДУЮ скобку и заменяет корневой объект
последней вложенной mapping. Это measurement_bug, не провал данных.

## Решение и минимум

Варианты: подавить предупреждения (не чинит смешанный stdout); исправить общий
decoder проход (выбран); читать отдельный service_report только в nightly
(оставляет ошибку mail-chain). Переиспользовать стандартный JSONDecoder и
existing helper, переходить за конец успешно разобранного объекта, не парсить
его внутренности повторно. Учитывать массивы/строки, чтобы их вложенные объекты
не выдавались за итоговый отчёт. Не менять status_from_payload или source gates.
Бюджет до25 добавленных обычных строк, новых файлов кода/зависимостей/флагов0.

Уточнение по независимому Lovelace review: отказ обязан вернуть непустой
status=error, иначе caller читает прежний expected_output. Не восстанавливаться
внутри malformed outer или после неоднозначного suffix; такой вывод STOP.
До первого JSON допустим warning, далее целые JSON-значения и пробелы.
Это закрывает также ложный PASS, не меняя status_from_payload. Тесты вне бюджета.
Первый Claude plan дал PASS; свежий pack нужен из-за изменения описания отказа
и inventory: заголовок «Границы» был исправлен в «СТОП» после прежнего scan.
Это ошибка порядка основного исполнителя, не изменение кода другим агентом.

## Приёмка

- Warning + pretty JSON с вложенными словарями сохраняет внешний статус pass.
- Несколько полных отчётов: последний внешний объект, не вложенный статус.
- Общий partial/blocked не становится ok из вложенного объекта pass.
- JSON строка/массив не становятся mapping отчётом из-за внутренних скобок.
- Повреждённый/отсутствующий итог не объявлять успешным.
- Реальный сохранённый log153345 парсится как сервисный report153346 с теми же
  статусами; ноль сетевых вызовов/записей БД для этой репродукции.
- Профильные тесты + независимый audit; commit/push/code-release существующим
  инструментом. Затем второй полный цикл. Первый service PASS сохранить как
  доказательство данных; ошибочную wrapper-квитанцию не переписывать задним числом.

## СТОП

Writer сейчас остановлен. D1 inventory/Claude/preflight до кода. Graphify wrapper
попытался обновить старую карту и остановлен sandbox; не повторять пересборку,
выводы подтверждать исходниками/git. Никаких prod/CRM/ASR/клиентских отправок,
нового seed, массовых выгрузок или ручного изменения activation/config.
