> DONE 2026-07-31 00:49 | ветка codex/calls-dialogue-m1-20260730 | codex

> TAKE 2026-07-31 00:31 | ветка codex/calls-dialogue-m1-20260730 | codex

Ветка: codex/calls-dialogue-m1-20260730
Зоны: src/mango_mvp/services/transcribe.py, src/mango_mvp/insights/pilot_extraction.py, src/mango_mvp/insights/llm_review.py, tests/test_dialogue_format.py, tests/test_pilot_extraction.py, tests/test_llm_review.py, docs/, tasks/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_dialogue_format.py tests/test_pilot_extraction.py tests/test_llm_review.py
Семантический-аудит: да

# ТЗ: потребительский гейт подтверждённых ролей

## Цель

До любого пилота сделать manager_quality_allowed реальным стоп-гейтом во всех
текущих путях оценки менеджера. Согласовать роли между двумя ASR, сбрасывать
устаревшее подтверждение после backfill и не выдавать неподтверждённые дорожки
за доказанные роли.

## Требования

1. При двух ASR подтверждать роль только при согласии двух вариантов.
2. Backfill сбрасывает старое подтверждение до полного повторного merge.
3. Mono, legacy, missing/false role gate и сложная топология не получают оценку.
4. Неподтверждённый стерео-результат публикуется с нейтральными дорожками.
5. Pilot extraction не создаёт score/missed-opportunity/LLM input без explicit true.
6. LLM review повторно проверяет explicit true и не восстанавливает запасную оценку.
7. Повреждённая карта физических каналов не переиспользуется молча.

## Приёмка

- Сквозные отрицательные тесты зелёные.
- Ни один forbidden case не получает число 0/55 вместо отсутствующей оценки.
- Старые данные без флага fail-closed.
- Независимый смысловой аудит PASS.
- Реальные ASR/R+A не запускались.

## СТОП

- Не запускать реальные ASR/R+A или внешние записи.
- Не разрешать оценку по умолчанию при отсутствии нового флага.
- Не менять клиентские/CRM-тексты и live-службы.

## Бритва

Один consumer gate и минимальные producer-дополнения; нетестовый код до 150
добавленных строк, без новых зависимостей и флагов.
