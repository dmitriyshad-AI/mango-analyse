> TAKE 2026-07-12 16:01 | ветка codex/tzv-calls-schedule-brand | codex

Ветка: codex/tzv-calls-schedule-brand
Зоны: scripts/, src/mango_mvp/customer_timeline/, tests/, docs/, tasks/, deploy/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_mango_calls_two_processes.py
Семантический-аудит: нет

# ТЗ-В. Звонки: раздельное расписание и бренд-метка

## Цель

Восстановить регулярную обработку новых звонков без повторения перегрузки ASR и без молчаливых сбоев.

## Требования

1. До включения установить причину остановки `com.mango.calls-two-processes` и выполнить подробную проверку окружения.
2. Переиспользовать `scripts/run_mango_calls_pipeline.py`, но запускать независимо:
   - Process A: Mango API -> существующий последовательный UI-пайплайн ASR -> Resolve -> Analyze -> drop;
   - Process B: drop -> timeline staging под single-writer lock.
3. Не запускать несколько внешних ASR-процессов. Допустима только внутренняя параллельность существующего `mlx_dual`.
4. У каждого процесса должны быть отдельные расписание, JSON-статус и курсор/водяной знак. Свежесть оценивается по данным, длительный простой даёт явный `stale`/`failed` с причиной.
5. Добавить детерминированную `brand_evidence`: `single`, `both` или `none`, без LLM. Значение и счётчики должны попадать в безопасный служебный отчёт.
6. Process A не пишет timeline. Process B не запускает Mango API/ASR/R+A. Production timeline, stable_runtime, AMO, Tallanto и CRM не трогать.
7. Сначала тесты и безопасные проверки. Службы включать только после независимого аудита каждого этапа.

## Приёмка

- две независимые launchd-задачи, без общего `cycle`-триггера;
- подробный preflight на отсутствующие Python/Codex/авторизацию/модули/секреты/диск;
- stale/fresh статусы основаны на фактических датах данных;
- тесты `single/both/none`, идемпотентность и lock;
- одновременно не работает больше одного внешнего ASR worker;
- отчёт и audit pack без ПДн.

## СТОП

- более одного внешнего ASR worker одновременно;
- путь Process B ведёт в production timeline или `stable_runtime`;
- Process B вызывает Mango API, ASR, Resolve или Analyze;
- бренд определяется моделью либо по `tenant_id`, а не по фактическому тексту;
- новая служба не сообщает конкретную причину сбоя.

## Контрольная точка 2026-07-12 18:06 MSK

- Код расписаний, подробного preflight, freshness и `brand_evidence` реализован; независимый аудитор дал PASS по этапам 0–2.
- Установлены две launchd-задачи: Process A каждые 1800 секунд, Process B каждые 900 секунд; legacy label не загружен.
- Process A завершён успешно: скачано 27 новых звонков, transcription done 268/268, analysis done 265/268, три прежних manual/pending сохранены; новый drop `quick_check=ok`.
- Внешние ASR worker шли строго последовательно: `transcribe`, затем `backfill-second-asr`; одновременных ASR worker не было.
- Исправлены реальные launchd-дефекты: тяжёлый import-preflight и отсутствие Homebrew `node` в PATH.
- Process B на новом drop запущен и в момент контрольной точки завершает импорт только в staging-копию. После него проверить итог, повтор `drop_unchanged`, полный pytest, финальный аудит и отчёт.
- Код SHA-курсора Process B считает фактический SHA ready DB; stale manifest не может дать false skip. Аудитор дал PASS.
