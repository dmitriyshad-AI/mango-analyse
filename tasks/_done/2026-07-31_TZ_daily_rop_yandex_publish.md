> DONE 2026-07-31 03:36 | ветка codex/calls-dialogue-m1-20260730 | codex

> TAKE 2026-07-31 02:32 | ветка codex/calls-dialogue-m1-20260730 | codex

Ветка: codex/calls-dialogue-m1-20260730
Зоны: scripts/export_daily_mango_calls_resolve.py, scripts/run_mango_calls_process.sh, tests/test_export_daily_mango_calls_resolve.py, tests/test_mango_calls_schedule.py, docs/, tasks/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_export_daily_mango_calls_resolve.py tests/test_mango_calls_schedule.py
Семантический-аудит: да

# ТЗ: ежедневный отчёт РОПа на Яндекс Диске

## Цель

После успешного Process B публиковать отчёт за завершённые московские сутки без дублей и без неподтверждённых ролей в основной управленческой выборке.

## Требования

1. Экспорт запускается после успешного Process B, только если в env задан каталог назначения.
2. Пути pipeline, Яндекс Диска, Tallanto и Mango переносимы между Mac.
3. В основной лист и статистику менеджеров допускаются только complete-звонки с `role_mapping.confirmed=true`, `manager_quality_allowed=true`, `topology=simple_two_party` и корректной хронологией.
4. Остальные звонки сохраняются полностью на листе проверки с понятной русской причиной.
5. Неизменный повтор переиспользует XLSX/TXT и не создаёт копий; изменившийся звонок обновляет пакет.
6. В именах файлов и техническом выводе нет телефона, email, ФИО и текста разговора.
7. Длинные расшифровки сохраняются полностью.

## Приёмка

- confirmed roles попадают в «Звонки»;
- timed dialogue без role evidence не попадает в «Звонки», но есть в «Проблемы данных»;
- повторный экспорт возвращает reused и не переписывает пакет;
- успешный Process B вызывает exporter, неуспешный не вызывает;
- без пути назначения старое поведение Process B сохраняется;
- целевые тесты, смысловой аудит и независимый breaker-аудит зелёные.

## СТОП

- Не запускать Process A/B, Mango API, Tallanto API, ASR или R+A на реальных данных.
- Не писать в реальный Яндекс Диск или Google Таблицу.

## Бритва

Расширить существующие exporter и wrapper; не создавать новую службу или зависимость. Новый механизм до 150 строк нетестового кода.
