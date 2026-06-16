# TZ-123 Target: вопрос вместо ухода на целевых кейсах

Дата: 2026-06-16  
Ветка: `codex/tz123-tz124-remeasure`  
Основа: TZ123 question layer + TZ124 anchored slot extraction.

## Что сделано

Собран целевой набор из 10 кейсов, где быстрый параметр реально не назван, от него зависит ответ, а OFF-плечо остаётся `draft_for_manager`.

Типы кейсов:

- цена без класса/формата/предмета: `сколько стоит у вас?`, `сколько стоит онлайн-обучение?`;
- расписание без предмета: `когда занятия?`, `по каким дням проходят занятия?`;
- скидки/группы без формата;
- подбор группы без времени;
- курс подготовки без предмета.

Новый воспроизводимый скрипт:

- `scripts/run_tz123_target_question_set.py`

Команда:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/run_tz123_target_question_set.py \
  --out-dir audits/_inbox/tz123_target_question_set_20260616 \
  --parallel 4
```

## Результат OFF → ON

| Метрика | Значение |
|---|---:|
| Кейсов | 10 |
| OFF: `draft_for_manager` | 10 |
| ON: `question_instead_of_handoff.fired` | 10 |
| ON: `bot_answer_self_for_pilot` | 10 |
| failed_checks | 0 |
| llm_calls_total | 0 |

Слоты:

| Слот | Срабатываний |
|---|---:|
| `grade` | 3 |
| `subject` | 3 |
| `format` | 2 |
| `time` | 2 |

## Semantic review

Verdict: `PASS_WITH_NOTES`.

Что прошло:

- каждый ON-ответ — один короткий человеческий вопрос;
- нет допроса: в каждом кейсе спрашивается ровно один параметр;
- P0 не перехвачен: все входы не P0, P0-лексика отсутствует;
- бренд не смешан: вопросы не называют чужой бренд;
- циклов нет: одноходовый target harness, повторов вопроса нет;
- OFF-плечо действительно показывает уход к менеджеру.

Неблокирующее замечание:

- Это целевой harness на входе слоя `draft_for_manager + answer_only`, а не доказательство прироста на полном Codex-direct replay. Полный replay после TZ124 ранее показал пустой актуальный остаток (`remainder_candidates=0`), потому что direct path уже отвечал сам.

## Артефакты

- Summary: `audits/_inbox/tz123_target_question_set_20260616/summary.json`
- Rows: `audits/_inbox/tz123_target_question_set_20260616/target_question_rows.jsonl`
- Human-readable transcripts: `audits/_inbox/tz123_target_question_set_20260616/transcripts.md`

`audits/_inbox/*` ignored, raw target artifacts не попали в git.

## Проверки

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_tz123_target_question_set.py
=> 1 passed

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q -k 'tz123_question or question_instead_of_handoff' tests/test_subscription_llm_draft_provider.py tests/test_telegram_dynamic_client_sim.py
=> 13 passed, 584 deselected

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_tz124_slot_anchor.py
=> 12 passed
```
