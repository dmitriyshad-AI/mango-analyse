> DONE 2026-09-06 06:39 | ветка codex/customer-timeline-source-gates-20260902 | codex

> TAKE 2026-09-06 05:37 | ветка codex/customer-timeline-source-gates-20260902 | codex

Ветка: codex/customer-timeline-source-gates-20260902
Зоны: src/mango_mvp/customer_timeline/wappi_history_import.py, src/mango_mvp/customer_timeline/nightly_service.py, scripts/run_customer_timeline_nightly_service.py, tests/test_wappi_history_import_to_timeline.py, tests/test_wappi_history_checkpoint.py, tests/test_customer_timeline_nightly_service.py, docs/DECISIONS_LOG.md, tasks/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_wappi_history_import_to_timeline.py tests/test_wappi_history_checkpoint.py tests/test_customer_timeline_nightly_service.py tests/test_customer_timeline_codex_task.py
Семантический-аудит: да
Feature-ID: feature.customer_timeline.stable_source_gates
Problem-ID: problem.customer_timeline.m4_first_cycle_source_failures
Исход: attempt_complete
Изменение: extend
Ключевые-символы: WappiPairCustomerResolver,prime_pair_chat_resolution,lookup_amo_opportunity_customers,validate_writer_ownership,run_nightly_service,service_config_fingerprint
Ключевые-слова: canonical alias identity,pending versus relink conflict,active M4 code release,immutable activation receipt

# R5: корректное сравнение личности и обновление версии writer

Продолжение активной цели владельца, исходный HEAD 2b4715af. FTS-фикс принят,
5751 тест пройден, 4 skipped. Не повторять завершённый перенос seed, скачанные
архивы, аудит всех веток и старые API-страницы. Рабочий r5 уже активирован на
c2704d1f и хранит доказанный частичный прогресс; его нельзя создавать заново.
## А. Точное сравнение личности Wappi

Переиспользовать WappiPairCustomerResolver и существующий read-only con в
from_store. Общий Store/read_api не содержит готового обхода alias-цепочек;
полный resolve_customer_identity_batches слишком широк и меняет сущности.

1. Одна локальная карта из customer_id_mappings текущего tenant, без LIMIT.
   Следовать только active alias с единственным направлением до существующего
   конечного клиента того же tenant. Active split, развилка, цикл, отсутствующая
   цель не должны незаметно исчезать из набора кандидатов или давать resolved.
   Не объединять людей по семье, телефону или похожему имени; mappings не менять.
2. Нормализовать владельцев ДО вычисления unique в pair, identity_owners,
   support и widget. Бренд-свидетельства старых/новых ID объединять множеством,
   не брать последнее. Нормализовать chat/exact owners и поздний lookup AMO.
   Слабые match_class, stoplist, split и brand-гейты не повышать.
3. prime_pair_chat_resolution создаёт конфликт владельцев только при ДВУХ
   resolved-ответах с разными каноническими ID. Unresolved widget вернуть как
   pending, не принимать вместо него pair и не добавлять строковый "None".
   Подтверждённый конфликт не терять при последующем чтении сообщений.
4. Проверить relink-границу existing/proposed после resolver. Только доказанная
   alias-эквивалентность разрешает одинаково сравнивать старый/новый ID.
   Opportunity другого физического владельца не обходить и не обнулять ради
   прохождения Store-check; до согласования принадлежности оставить pending.
5. Счётчик нарушений не должен считать отсутствие попытки перепривязки:
   existing pending + proposed empty остаётся прежним карантином, без нового
   blocked_customer_relink_conflicts. Если предложен другой клиент, сохраняются
   прежние запреты и счётчики. Это measurement_bug, не разрешение публиковать
   реальные противоречия. Никакие строки/пары вручную не удалять.

Приёмка А: chain A->B->C, split+alias, цикл, чужой tenant, missing target,
weak evidence, два ребёнка, два resolved разных владельца, pending->pending,
старый opportunity owner. Искусственная выборка и затем read-only replay по
сохранённому локальному кэшу; в отчёт агрегаты и эффективная видимость карантина.
Исторический ориентир 13 alias-only/28 ambiguous из 41 не подгонять под PASS.
Остаточный настоящий конфликт не объявлять разрешённым без нового доказательства.

## Б. Законный code-release уже активного M4

Три варианта: переписать ownership/activation (запрещено); снять HEAD-check
(запрещено); неизменяемая release-квитанция поверх прежней передачи (выбрано).
Это необходимое дополнение существующего validator/CLI, не второй writer,
не новый перенос M1, не пакет публикации production.

1. В существующем CLI добавить отдельную команду
   --approve-code-release PREVIOUS_SHA NEW_HEAD --review-receipt PATH.
   Она НЕ запускает nightly, не открывает SQLite на запись и не создаёт Store.
2. Сначала штатные lock с timeout=0 в прежнем порядке: wrapper run-lock,
   service run-lock, DB writer-lock. Конфиг/квитанции перечитать после locks.
   Занято -> STOP без сигналов чужим процессам. Переиспользовать существующие
   customer_timeline_run_lock/customer_timeline_writer_lock и atomic write_json.
3. Сохранить все прежние stop/ownership/activation и stable-config проверки.
   Разделить внутреннюю проверку исторических квитанций и проверку текущей
   разрешённой вершины кода. Обычный run не получает bypass/skip-check аргументов.
4. NEW_HEAD должен быть текущим чистым HEAD и потомком PREVIOUS_SHA. Проверить
   отсутствие незакоммиченных исполняемых файлов в src/scripts, не чистить data.
   Пути DB/state/worktree канонические; symlink-подмена квитанций запрещена.
   Review проверяется штатным make_audit_pack --verify-receipt, привязан к
   окончательному HEAD с полным переходом PREVIOUS->NEW, включая этот CLI.
   Дополнительно manifest.head == NEW_HEAD, branch_diff_base == полный
   PREVIOUS_SHA, allowlist покрывает изменённые кодовые файлы этого перехода.
   Старый preflight/одно поле PASS недостаточны; исторические review не
   перевалидировать относительно будущих HEAD.
5. По одному неизменяемому файлу state/WRITER_CODE_RELEASE_<PREVIOUS_SHA>.json.
   Поля: schema, previous/new SHA, SHA предыдущей квитанции, SHA исходных
   ownership/activation, canonical DB/worktree, stable-config SHA, review path
   и SHA, время утверждения. Первый переход опирается на activation.
   Validator проходит цепь от activation, проверяет связи/хеши/пути/циклы и
   требует текущий HEAD равным вершине в новом входе запуска. Старые checkout
   не знают нового validator: их прямой запуск запрещён эксплуатационным
   правилом, старое расписание остаётся отключённым. Это не машинный запрет
   произвольного запуска исторического кода; внешний контроллер не строить.
   Посторонние/оторванные release-файлы нельзя молча считать разрешением.
6. Запись атомарна, каталог синхронизируется. Совпадающий повтор возвращает
   существующую квитанцию без изменения времени/байтов. Конфликтующий повтор
   запрещён. Crash до публикации оставляет старую вершину, после неё разрешена
   новая; исторические файлы неизменны. Отдельный current.json не нужен:
   имя исходящего перехода задаёт единственную цепь.
7. Эффективные code/release SHA включить в fingerprint ВОЗОБНОВЛЕНИЯ run,
   не в stable ownership fingerprint. Новый код не продолжает старый run как
   тот же запуск, но переиспользует независимые checkpoints источников.
   Эффективный fingerprint вычислить под service-lock после проверки цепи.
   В отчёте различать исходную activation и текущую рабочую ревизию.

Приёмка Б: реальный CLI на synthetic state; повтор; чужой/грязный HEAD;
stale review; изменённый путь/конфиг; цикл/повреждённая цепь; concurrent
approve/run; crash вокруг атомарной записи; старый HEAD в новом validator;
запрет resume между версиями. Исторические квитанции и bytes БД неизменны.

## Объём и процесс

Не создавать новый framework/таблицы/зависимости/фоновые службы. Два узких
дополнения текущих владельцев: нормализация идентификаторов и code-release.
Ориентир до 150 добавленных нетестовых строк на новый механизм; safety-проверки
не сокращать ради бюджета. При превышении сначала более узкие варианты.
Один canonical audit pack. Сначала inventory и Claude preflight, затем код,
тесты и независимые роли. Полный pytest один раз после обоих исправлений.
На окончательном коммите отдельный read-only release review перед approve.

## Приёмка цели после кода

Законный approve на остановленном r5; продолжить AMO со страниц 201 и Wappi с
сохранённого checkpoint (3 incremental профиля, 1 незавершённый audit).
Два последовательных overall_status=ok AND data_quality_status=pass; один
compact и существующие S100/S10/реальный reader; только затем одно расписание
mail-chain && nightly-warehouse. Если source-гейт выявил реальное противоречие,
не называть цикл зелёным и не ослаблять gate. Calls upstream отмечать отдельно.

## СТОП

Нельзя переписать историю передачи, подменить review, предположительно связать
семью, убрать настоящий конфликт или писать в prod/CRM/Tallanto/Wappi.
Никаких ASR/массовых LLM/клиентских сообщений. ПДн только локально; Claude
получает проверенные код/docs/агрегаты, не строки клиентов. Расписания до
полной приёмки остаются выключены. Preflight оценивает ПЛАН, не наличие ещё
не реализованной функциональности на исходном HEAD.

Диагностика preflight: PII-detector ошибочно распознал часть hex
status_fingerprint как телефон. Подтверждено структурным чтением JSON:
затронут только sha256, клиентских значений нет. Этот ложный сигнал сохраняем
в отчёте; настоящий ПДн-гейт не отключать и receipt вручную не создавать.
