> TAKE 2026-09-06 17:33 | ветка codex/customer-timeline-source-gates-20260902 | codex

Ветка: codex/customer-timeline-source-gates-20260902
Зоны: src/mango_mvp/customer_timeline/amo_incremental.py, tests/test_customer_timeline_amo_incremental.py, docs/DECISIONS_LOG.md, docs/worktrees_registry.md, tasks/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_customer_timeline_amo_incremental.py tests/test_customer_timeline_nightly_service.py tests/test_customer_timeline_contracts.py
Семантический-аудит: да
Feature-ID: feature.customer_timeline.source_pagination_completion
Problem-ID: problem.customer_timeline.m4_first_cycle_source_failures
Изменение: fix
Ключевые-символы: fetch_endpoint_checkpointed,_fetch_amo_time_windows,save_amo_incremental_checkpoint
Ключевые-слова: fresh leads windowed fetch,repeated pagination drift,unverified checkpoint retention

# AMO: завершать нестабильную дельту внутри одного source-вызова

## Факт и причина

HEADe177f0ea. Полныйrun132441закончилсяpartialза2102.738s толькоиз-заleads
pagination_universe_changed. Штатное source-only recovery затемPASS373.931s,
3остальныхendpoint0GET,leads171GET/87страниц/3окна. Stdoutсохранён в
audits/_inbox/customer_timeline_r5_source_completion_20260906/AMO_RECOVERY_STDOUT_EVIDENCE.md.
Глобальныйreportуспелротироваться, его поздняякопияНЕявляетсяPASS-evidence.
Новый полныйrun141027сноваpartialleads129строк/7страниц/1непрошедшаяпроверка,
остальные3endpointcomplete. Нельзя повторять source-only->full как решение:
fresh-fetch вновь начинает обычные страницы и использует окна лишь со следующего
вызова. Конкретные изменившиесяполяAMOнеустановлены; наблюдение=anchor mismatch.

АрхитекторAmpereпрочиталкод/отчёты:existingwindow-loop можетвосстановитьсявтомже
вызове. Обнаружено также: window_pending_items перезаписывается/очищается,
поэтому недоказаннаястрокаисчезаетизcheckpoint;D153обещал её сохранение.

## Минимальное решение

Варианты: повторять весь цикл (лишние 35 минут); inline fallback после
доказанного дрейфа; сразу направить новый leads в окна. Выбран inline fallback.
Первая рекомендация fresh-leads -> windows ОТОЗВАНА: стабильный плотный backlog
с одинаковой секундой и количеством страниц больше бюджета перестал бы
завершаться. Ampere независимо воспроизвёл: обычный путь завершается, оконный
не продвигается. Исходный интеграционный тест 1000 строк оставить без изменений;
добавить отдельный тест действительно одинаковых timestamp (в текущем fixture
они увеличиваются с индексом). Это причина повторного plan-review, не дубль.
Не новый APIclient, importer, scheduler, флаг или схема.

1. Complete-checkpoint остаётся самымраннимвозвратом,0GET.
2. Только при drift и остатке бюджета >0 вызвать existing window helper в этом
   вызове после сохранения drift-checkpoint. Не менять стабильную пагинацию.
   Передать остаток N-batch_pages, общий requests, фиксированный upper_bound.
   Сложить pages_this_run/verification_pages/fetched_this_run, но НЕ pages
   (накоплено helper). Итоговый статус из восстановления; исходный drift только
   диагностический. При нулевом остатке оставить текущий non-PASS checkpoint.
3. max_pages<=0нормализоватьпоexistingконтрактуN=1; общийбюджетNстраницданных,
   не болееNпроверок,+2anchorprobes,включаяявные429retriesв2N+2GET.
4. Недоказанные payload сохранять в existing unverified_items с дедупом версий
   по stable_digest, не включать в подтверждённые items/DB. Не терять при сужении
   или успехе окна. После завершения сохранить последний проблемный цикл каждого
   endpoint в служебном ключе существующего private checkpoint; стабильные циклы
   сохраняют эту диагностику, следующий инцидент заменяет предыдущий. Не новая
   БД/файл/очередь применения; без бесконечного архива всех циклов. Полные версии,
   а не 20 хешей (хеш не позволяет разобрать потерянное сырьё). Граница/cursor/
   счётчики фактов от диагностического ключа не зависят. Не называть page_cap
   новым drift. Сохранять состояние при восстановлении/исключении/смене fingerprint.
5. Бюджетобычногокода<=50добавленныхстрок,safety/testsсчитатьчестноотдельно.
   Единственныйисполнительэтойветки;доокончанияexec54372код/HEADНЕменять.

## Приёмка

- drift->complete одним вызовом при достаточном бюджете, строки не теряются;
- стабильный плотный backlog продолжает постраничное чтение через несколько
  вызовов; исходные проверки не переносить на другой endpoint ради зелёного;
- неизменныйupper_bound,исчерпаниебюджета,429,max_pages0;
- исчезнувшая строка остаётся unverified и НЕ становится verified/фактом клиента;
  диагностика переживает успешное применение и следующий стабильный цикл;
- триcompletecontacts/events/tasks=0GET; resumeстарыхcheckpointнеухудшен;
- стабильноепервоеокно:нетложнойдиагностикиdrift;
- послеформальныхтестов независимыйClaude/codeаудит;releaseквитанцияпо
  существующейпроцедуре,непереписыватьстаруюактивацию/конфигручнымпутём;
- затемдвастрогихполныхцикла0дублей,overallok/data_qualitypass. Source-only
  илиtestPASSнепереименовыватьвдостигнутуюцель. Callsfreshnessgapотдельно.

## СТОП

Inventory/Claude/preflight по D1 до кода. Существующееcallers/тестынеослаблять.
Нольprod/CRM/Tallanto-write/ASR/клиентскихотправок. Не новыйseed/копияБД.
Не менять активныйwriter. Сохранять reportДОследующеговызоваисточника,
команды/решениявWORKFLOW_JOURNALдлябудущегопроверенногоskill.
