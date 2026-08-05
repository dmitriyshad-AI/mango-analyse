> DONE 2026-08-05 05:06 | ветка main | codex

> TAKE 2026-08-05 03:50 | ветка main | codex

Ветка: main
Зоны: src/mango_mvp/customer_timeline/store.py, src/mango_mvp/customer_timeline/tallanto_attendance_import.py, src/mango_mvp/customer_timeline/wappi_history_import.py, tests/test_customer_timeline_store.py, tests/test_customer_timeline_tallanto_attendance_import.py, tests/test_wappi_history_import_to_timeline.py, tasks/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_customer_timeline_store.py tests/test_customer_timeline_tallanto_attendance_import.py tests/test_wappi_history_import_to_timeline.py
Семантический-аудит: да

# ТЗ: отзывать старые события при новом конфликте владельца

## Проблема

Свежий read-only canary Tallanto attendance на staging нашёл один точный Tallanto ID,
который теперь корректно заблокирован конфликтом идентификации. При этом за прежним
клиентом остаются восемь активных `tallanto_attendance_api` событий, все с
`writeoff_confirmed=true`.

Память бота сейчас закрыта адресным конфликтным гейтом, но менеджерская хронология
читает эти активные события напрямую. После закрытия конфликта устаревшие события
также смогут снова повлиять на сводку. Это дефект жизненного цикла: входной гейт
останавливает новую запись, но не отзывает старую атрибуцию.

## Образ результата и бизнес-польза

Если точный владелец события стал спорным, факт больше не относится ни к одному
клиенту до разрешения конфликта. Менеджер не видит чужое посещение или списание,
бот не получает его после снятия временного гейта, а исходное событие и аудит не
удаляются. Повторный проход ничего не меняет.

## Минимальное решение

1. В `CustomerTimelineSQLiteStore` создать один общий метод карантина всех активных
   событий по `tenant_id + source_system + source_id`; пара не обязана быть уникальной.
2. Метод обязан переиспользовать `_retire_dependencies`, снять владельца и
   opportunity, поставить `match_status=ambiguous`, `confidence=0`,
   `metadata.pending_attribution=true`, запретить bot-context, отозвать активную
   bot-safe summary прежнего клиента и оставить audit trail.
3. Заменить реализацию `quarantine_conflicting_wappi_events` вызовами общего метода;
   не держать второй алгоритм в Wappi.
4. В attendance apply-пути перед записью новых событий передать в общий метод только
   активные события штатного `tallanto_attendance_api`, чей Tallanto ID реально вошёл
   в конфликт текущего incremental-окна. Старый файловый источник требует отдельного
   replay-safe плана восстановления и в этом блоке не меняется.
5. Не добавлять новый фильтр чтения, feature flag, зависимость или файл кода.

## Критерии готовности

- существующее attendance-событие после появления точного конфликта имеет
  `customer_id IS NULL`, `pending_attribution=true`, `match_status=ambiguous`;
- менеджерская хронология прежнего клиента больше не содержит этот факт;
- связанные derived signals/chunks и bot-safe summary прежнего клиента отозваны;
- повторный apply даёт ноль новых карантинов и не меняет `record_hash`;
- разрешённый точный ID и `tallanto_identity_ambiguous` с одним кандидатом не
  карантинятся;
- Wappi сохраняет прежнее бизнес-поведение через тот же общий метод;
- при отсутствующей строке Wappi-события сводка известного прежнего клиента всё равно
  отзывается; все события одной source-пары карантинятся, а не только первое;
- `COUNT(*)` событий не уменьшается, исходное `record_json` остаётся доступным;
- canary на APFS-клоне: quick_check=ok, FK=0, конфликтный старый факт не принадлежит
  клиенту, 15 разрешённых replay-ID по-прежнему принадлежат правильным владельцам;
- никаких записей в product Timeline, Tallanto, AMO, CRM или Wappi.

## Приёмка

Целевые тесты проходят. Затем один apply-проход и повтор выполняются только на
APFS-клоне staging-БД. В отчёте отдельно фиксируются `formal_pass`, `data_pass`,
`semantic_pass`, `business_pass` и `runtime_pass`; один общий PASS запрещён.

## СТОП

- найденный общий метод не может сохранить аудит и исходное событие;
- для исправления требуется запись в product/runtime БД или внешнюю систему;
- разрешённый точный владелец теряет событие либо число событий уменьшается;
- изменение требует нового эвристического правила идентификации;
- целевые тесты или quick_check/FK не проходят.

## Бритва

Добавление общего метода допустимо только при одновременном удалении поглощённой
Wappi-реализации. Цель нетестового diff: не более +80 строк и отрицательный баланс
строк либо явное объяснение, почему он временно положительный.
