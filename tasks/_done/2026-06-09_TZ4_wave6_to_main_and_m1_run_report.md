# TZ4 Wave 6 To Main And M1 Run Report

Дата: 2026-06-10

## Итог

- Волна 6 перенесена в `main` отдельным коммитом `8b48f196`.
- Флаг `TELEGRAM_LLM_RETRIEVE` остаётся default OFF.
- Перенос сделан cherry-pick только коммита Волны 6 `7389d07a`, без отката более новых коммитов `main` TZ1-TZ3.
- Бандл: `/Users/dmitrijfabarisov/Yandex.Disk.localized/OpenClaw/Actual Mango Tests/mango_clean_8b48f196`.
- M1-задачи smoke89:
  - `/Users/dmitrijfabarisov/Yandex.Disk.localized/OpenClaw/Actual Mango Tests/tasks/_inbox_m1/2026-06-10_wave6_base_smoke89_codex.task.yaml`
  - `/Users/dmitrijfabarisov/Yandex.Disk.localized/OpenClaw/Actual Mango Tests/tasks/_inbox_m1/2026-06-10_wave6_on_smoke89_codex.task.yaml`

## Что изменено

- `src/mango_mvp/channels/subscription_llm.py`
  - добавлен LLM-ретривер фактов прямого пути за `TELEGRAM_LLM_RETRIEVE`;
  - кандидаты фильтруются детерминированно по active_brand и client_safe до LLM-вызова;
  - ретривер выбирает только id фактов, клиентский текст не генерирует;
  - invalid/hallucinated id отбрасываются;
  - timeout/ошибка/пустой выбор fail-soft возвращают keyword pack;
  - при ON P0-преблок выполняется до ретривера и direct-draft.
- `scripts/run_telegram_dynamic_client_sim.py`
  - добавлен счётчик роли `bot_retriever` в `llm_calls`.
- `tests/test_subscription_llm_draft_provider.py`
  - NEG: OFF-паритет;
  - NEG: релевантный enrollment-факт;
  - NEG: бренд-изоляция до модели;
  - NEG: fail-soft timeout;
  - NEG: hallucinated id отбрасывается;
  - NEG: только hallucinated id -> keyword fallback;
  - NEG: P0-преблок не вызывает ретривер и direct draft.
- `tests/test_telegram_dynamic_client_sim.py`
  - счётчик `bot_retriever` в summary.

## Проверки

- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_subscription_llm_draft_provider.py -k 'wave6_llm_retrieve'`
  - `7 passed`
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_telegram_dynamic_client_sim.py -k 'retriever or llm_call_summary'`
  - `2 passed`
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests`
  - `2856 passed, 2 skipped, 1 warning`

## M1

- Оба task.yaml используют один бандл `mango_clean_8b48f196`.
- Набор: `smoke_v2_acceptance_2026-06-08.jsonl`.
- SHA набора: `bb9c7001c92ba36e8475ba8a8e4ef4c5621e3dc84099fd1bd837940a285efda0`.
- Судья: `v9`.
- `parallel: 4`.
- Разница:
  - base: `TELEGRAM_LLM_RETRIEVE: "0"`;
  - on: `TELEGRAM_LLM_RETRIEVE: "1"`.

## Не трогал

- `CLAUDE.md` в рабочем дереве был изменён до этой задачи; в коммит TZ4 не включён.
- Старые subset30-задачи Волны 6 в Yandex inbox не удалялись.
