> TAKE 2026-08-15 11:35 | ветка codex/google-publisher-20260815 | codex

Ветка: codex/google-publisher-20260815
Зоны: scripts/publish_live_mango_calls_google.py, tests/test_publish_live_mango_calls_google.py, docs/, tasks/, audits/_inbox/, внешний owner-only runtime publisher только после всех GO-гейтов
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. python3 -m pytest -q tests/test_publish_current_mango_calls_google.py tests/test_publish_live_mango_calls_google.py
Семантический-аудит: да

# ТЗ: заменить рабочий Google publisher без остановки обработки звонков

Статус: разрешено Дмитрием 2026-08-15. Меняется только publisher рабочего листа.

## Цель

Заменить медленную natural-language автоматизацию одним проверяемым скриптом из
репозитория. Capture, Whisper, GigaAM, Resolve и Analyse продолжают работать.

## Обязательные инварианты

- Ровно один writer рабочего листа; старый и новый не пишут одновременно.
- Рабочая SQLite остаётся единственной базой звонков. Publication ledger —
  отдельный атомарный owner-only JSON sidecar без ПДн и текста; таблицы и схему
  рабочей SQLite publisher не меняет.
- Перед записью — полная однозначная сверка A:P с SQLite. Неизвестная или
  неоднозначная физическая строка блокирует Google write.
- Reservation создаётся до Google write; потерянный ответ сначала разрешается
  полным readback, слепой повтор append запрещён.
- Все изменения листа выполняются одним `spreadsheets.batchUpdate`: A:P,
  временный Q, сортировка A:Q, очистка Q и layout.
- После exact readback и повторной проверки source fingerprint сначала
  атомарно фиксируется `verified` в sidecar, затем короткой транзакцией
  совместимый `sync_status=done`. Сбой между этими шагами восстанавливается из
  verified sidecar без повторной Google-записи.
- Ровно 16 колонок A:P; МСК; длительность `N мин M с`; newest-first; Q:Y пусты.
- J = WRAP/TOP, P = CLIP/TOP; высота строки считается только из J.
- Полная расшифровка обязательна; формульная защита для каждого текстового поля.
- Ячейка длиннее 50 000 Unicode-символов не обрезается: конкретная строка
  получает явную data-error и не блокирует другие валидные строки.
- Публикатор не меняет ASR/Resolve/Analyse payload и не очищает их claims/leases.

## Переиспользование

Переиспользовать защищённое чтение service-account credentials, Google session,
flock и REST transport из `scripts/publish_current_mango_calls_google.py`.
Не переиспользовать его старую 19-колоночную бизнес-схему.

## Проверки до live

1. Fake Sheets: success, timeout/lost response, crash до/после write/readback,
   параллельные writers, дубль/ambiguity, stale Analyse, DB busy, invalid data.
2. Точное форматирование 16 колонок, UTC→МСК, округление x.5 вверх, сортировка,
   Q:Y empty, J/P и высоты только по J.
3. Read-only shadow на полном живом листе: 100% identity, 0 duplicate/missing,
   zero-write повтор, измеренная производительность.
4. Независимый code/design audit Codex и Claude CLI: P0/P1 = 0.

## Реализованный снимок до cutover

- Новый writer оформлен одним checked-in скриптом и отдельным owner-only
  sidecar; рабочая SQLite и код Capture/ASR/Resolve/Analyse не изменены.
- Целевой набор: 98 тестов; отдельно пройдены `py_compile` и
  `git diff --check`.
- Lost-response, crash/recovery, DB busy, два writer, stale source, формулы,
  неверная вкладка, неоднозначная identity, legacy-даты/телефоны, UTC→МСК,
  длительность, сортировка, Q:Y и физические высоты покрыты тестами.
- Fake Sheets хранит высоты отдельно от данных строк и применяет реальные
  `updateDimensionProperties`; дополнительный случайный аудит проверил 29 000
  комбинаций целевых/текущих высот без расхождений.
- Claude CLI выполнил три read-only прохода. Первый нашёл три P1: неполный
  shadow layout/order, 60-секундный write-timeout и слабую диагностику строки.
  Все замечания исправлены. Повторный и финальный проходы дали `GO`, новых
  P0/P1 нет.
- Независимые Codex-аудиты архитектора и ломателя после тех же исправлений:
  `P0=0`, `P1=0`, `GO`.
- Последний согласованный read-only shadow до финального cutover сопоставил
  100% физических строк без ambiguity; 15 непроецируемых Analyse-пayload
  изолированы как явные data-error и не могут получить ложный `sync done`.
  После паузы старого writer этот гейт выполняется повторно на свежем снимке.

## Cutover

1. Дождаться завершения текущего старого Google-run.
2. Остановить только старый writer и доказать отсутствие выполняющегося run.
3. Bootstrap ledger из точного readback без изменения Google.
4. Канарейка до 25 строк; полный readback; повтор без изменений = 0 writes.
5. Включить только новый publisher; Capture/ASR/Resolve/Analyse не трогать.

## Приёмка (GO)

- 0 дублей, пропусков, cross-call и ambiguous identity;
- 100% Google writes подтверждены точным readback;
- newest-first, МСК, duration, 16 headers, Q:Y empty, layout соблюдены;
- очередь publisher убывает, скорость выше устойчивой скорости Analyse;
- обработка звонков не прерывалась.

## СТОП и откат

При двух writers, дубле, неясной identity, ложном verified, остатке Q, неверном
layout или росте блокировок остановить только новый publisher. Сохранить ledger,
сделать read-only reconcile и выполнить fix-forward; конвейер звонков не трогать.

## Независимый аудит Claude CLI до реализации

Claude дал `REVISE`. Приняты его замечания: sidecar вместо новой таблицы в
рабочей SQLite; отдельное восстановление между verified-sidecar и sync-status;
жёсткий лимит 50 000 символов; ровно один новый скрипт без нового пакета и
installer. Заголовки A:P уже заморожены живым листом и требованиями Дмитрия,
поэтому замечание об их отсутствии отклонено. Глобальный Q-sort и высоты после
sort сохраняются в одном batch: живой замер показывает секунды на сам Google
batch, а не десятки минут; это проще и точнее targeted insert для stale-строк.
