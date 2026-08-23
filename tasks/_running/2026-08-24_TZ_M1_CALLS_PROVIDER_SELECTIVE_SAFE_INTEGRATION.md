> TAKE 2026-08-24 00:30 | ветка codex/m1-calls-provider-selective-integration-20260824 | codex

# ТЗ: безопасно интегрировать provider roles и Selective Resolve на M1

Ветка: codex/m1-calls-provider-selective-integration-20260824
Зоны: src/mango_mvp/, scripts/, tests/, docs/, deploy/, product_data/, .agents/, .claude/, AGENTS.md, ARCHITECTURE.md, CLAUDE.md, README.md, tasks/, audits/_inbox/m1_calls_provider_selective_integration_20260824/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_mango_calls_two_processes.py tests/test_analyze.py tests/test_non_conversation_quality.py tests/test_resolve.py tests/test_ingest_filename_parse.py tests/test_productization_mango_office_client.py tests/test_publish_daily_mango_calls_google.py tests/test_deal_aware_stage1_snapshot.py tests/test_amocrm_deals.py
Семантический-аудит: да
Feature-ID: feature.calls.provider_roles_selective_resolve
Problem-ID: problem.calls.provider_evidence_missing_and_unverified_selective
Изменение: extend
Ключевые-символы: capture_provider_role_evidence,fetch_recording_transcripts,_semantic_selective_input,PROVIDER_EVIDENCE_FIELD

Дата: 2026-08-24.

## Цель простыми словами

Перенести уже проверенный пилот M4 в отдельную офлайн-ветку M1, не создавая
второй Mango-клиент, второй publisher, новую БД или службу. До включения
устранить четыре подтверждённых риска: потерю коммерчески важных сервисных
звонков из контроля РОПа, ложное убийство долгого живого Whisper, подвешенные
claims при остановке процесса и падение всей партии из-за позднего дубля
`source_recording_id`.

## Источники

- База ветки: текущий `origin/main` на момент создания этой попытки.
- Донор пилота: `codex/calls-provider-selective-pilot-20260822`.
- Полный SHA донора проверяется Git и пакетом передачи перед переносом.
- Донор watchdog: ветка `codex/pipeline-stall-watchdog-20260818`; переносить
  только доказанную безопасную дельту, не коммит целиком.
- Пакет: `OpenClaw/M1_calls_provider_selective_handoff_20260823_v1`.

Проверка пакета уже должна подтвердить все SHA, исправный bundle, `quick_check=ok`
и 6 383 строки. Корпус остаётся вне Git и используется только read-only либо в
отдельной временной копии для миграции.

## Обязательный порядок

1. Новый inventory получает вход `Изменение: extend`. Только inventory решает,
   что точная донорская реализация требует `port`.
2. Новый Claude context pack и read-only Claude Opus создаются до переноса.
3. Только после `PREFLIGHT: OK` переносится пилот. Заблокированная попытка от
   23 августа не является базой и не сливается.
4. Конфликты читаются с обеих сторон. Процесс разработки, реестр M1 и решения
   текущего main сохраняются; решения пилота перенумеровываются после уже
   занятых D-115…D-119.
5. Два WIP-коммита selective GigaAM v2 и раннего batch GigaAM остаются в своих
   ветках и в интеграцию не попадают.

## Инварианты provider roles / Selective Resolve

- Используется существующий `MangoOfficeClient.fetch_recording_transcripts`.
- Allowlist Mango не больше 10 записей.
- Любой ответ Mango кроме `result=1000` даёт fail-closed.
- Проверяются `source_call_id`, `source_recording_id` и SHA записи.
- Provider evidence проходит capture → ingest → Transcribe → Resolve.
- Без доверенных ролей модель не вызывается.
- Selective Resolve и все новые режимы по умолчанию выключены.
- Существующие 18 столбцов Google и UTM переиспользуются без второго publisher.
- Рабочая SQLite и исходный корпус не открываются через мигрирующую фабрику.

## Риск 1. Коммерчески важный сервисный звонок

Тип звонка нельзя грубо менять на `sales_call`. Добавить отдельный
детерминированный флаг коммерческой проверки только для финальных non-sales
типов. Он должен:

- ставить `needs_review=true` и понятную русскую причину РОПу;
- сохраняться в `analysis_json.quality_flags`, не ломая строгий контракт
  `structured_fields.commercial`;
- показываться существующим Google publisher без изменения схемы 18 столбцов;
- считаться отдельно от `sales_calls` в stage1;
- не давать сервисному звонку штраф при подборе AMO-сделки, но и не давать
  бонус продажи;
- блокировать автоматический writeback до ручной проверки.

Положительные сценарии: обещанный демоурок не получен; обсуждение курса с
согласованным продолжением; неполная оплата и обещанный остаток; ожидаемая
оплата вместе с риском отказа. Отрицательные: повторная пересылка документов,
перевод между сменами, полностью поступившая оплата при оставшемся договоре,
чистая логистика без следующего коммерческого действия. Для всех сценариев
исходный `call_type` не меняется и модели не вызываются.

## Риск 2. Долгий живой Whisper

- Общий лимит стадии четыре часа сохраняется.
- Watchdog по отсутствию роста лога выключен по умолчанию и включается только
  явной настройкой; значение либо `0`, либо не меньше 60 секунд.
- Долгий живой Whisper нельзя убивать только из-за прошедшего времени.
- При включённом watchdog рост лога продлевает дедлайн, настоящий stall
  завершает всю process group.

## Риск 3. SIGTERM и claims

- SIGTERM-handler работает для всего последовательного pipeline, а не только
  parallel-ASR.
- Родитель задаёт строгий `MANGO_CALLS_STAGE_WORKER_ID`; все стадии используют
  один и тот же точный идентификатор.
- Пока процесс жив, heartbeat продлевает только claims этого worker.
- После полного завершения process group при timeout, SIGTERM или crash только
  claims этого worker атомарно возвращаются в `pending`.
- Recovery работает узким SQL без запуска миграций. Ошибка recovery блокирует
  следующую стадию и не маскируется успешным shutdown.

## Риск 4. Поздний дубль записи

- Перед вставкой повторно проверяется нормализованный `source_recording_id`.
- При доказанном конфликте доверие снимается только с новой строки:
  `source_recording_id=None`, отдельный счётчик конфликта увеличивается.
- Гонка на unique index обрабатывается savepoint и повторной проверкой.
- Недоказанная ошибка БД по-прежнему fail-fast.
- Здоровые строки той же партии импортируются.

## Проверки корпуса и миграции

1. Воспроизвести на read-only корпусе: 6 383 проверено, 24 изменения типа,
   16 `service_call → non_conversation`, 8 `technical_call → service_call`,
   без ASR, Resolve, Analyze и моделей.
2. Воспроизвести полный сохранённый контекст: 4 966 строк, manual review
   828 → 1 013, добавлено 190, снято 5.
3. Проверить все 24 строки бизнес-аудитором и не терять продажи.
4. `init-db` запускать только на второй временной копии корпуса: колонка и
   уникальный индекс появляются один раз, повторный запуск пустой, исходный SHA
   не меняется.

## Тесты

- Канонический семифайловый контур из пакета после переноса.
- Тесты provider roles, Selective Resolve, publisher, stage1, AMO и late duplicate.
- Тесты default-off watchdog, opt-in stall, log growth, обычного и parallel
  SIGTERM, heartbeat, точного recovery и блокировки следующей стадии при ошибке.
- Полная фактическая команда и вывод сохраняются.
- Все тесты обязательно повторяются на Python 3.12, который использует служба.
  Результат только на Python 3.14 не является приёмкой.

## Независимый аудит

После реализации сформировать новый замороженный пакет и вызвать Claude Opus в
ролях архитектора, ломателя, бизнес-аудитора и уборщика. Принятые замечания
закрыть тестом, гейтом или явным ручным контролем; после изменений пакет и
receipt пересобрать.

## Приёмка

1. Inventory доказал `port` из точного донорского кода, preflight до изменений
   дал `OK`.
2. Пилот перенесён без отката нового процесса main и без двух WIP GigaAM.
3. Все четыре риска закрыты тестами и локальными доказательствами.
4. Корпусные числа воспроизведены, исходная SQLite не изменилась.
5. Точный тестовый контур зелёный на Python 3.12.
6. Финальный Claude-аудит дал `PASS`, ветка чистая и имеет точный SHA.

## СТОП

- Любое включение launchd/production или изменение live-worktree.
- Любое обращение к Mango API, пока отдельно не снята ошибка `5008`.
- ASR, реальный Resolve/Analyze, модельные вызовы и внешние записи.
- Изменение исходного корпуса или рабочей SQLite.
- Inventory `stop`, preflight без receipt, потеря доверия provider evidence,
  потеря продажи, неустранённый подвешенный claim или падение здоровой партии.
