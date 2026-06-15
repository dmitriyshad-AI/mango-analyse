# TZ-117: трассировка ошибок блоков A/B/C/D/E

Дата: 2026-06-15

## Статус

Выполнено. Файл `2026-06-15_TZ117_trassirovka_oshibok_blokov.md` в рабочих деревьях не найден, реализация сделана по тексту ТЗ из чата Дмитрия.

## Что изменено

- Добавлен генератор единых trace-файлов: `scripts/build_tz117_error_traces.py`.
- Для D добавлено явное поле `rationale` в тот же Codex-вызов назначения ролей; версия кэша поднята до `mono_role_assignment_v2`.
- В D offline runner добавлена колонка `codex_rationale`.
- Добавлены NEG/регрессионные тесты на подсчёт `model_fix/model_break/both_wrong`, маскирование email/телефонов и корректную трактовку B/E review verdict.

## Артефакты

Trace pack:

`audits/_inbox/tz117_error_traces_20260615_232756/`

Внутри для каждого блока есть:

- `*_trace.csv`
- `*_trace.jsonl`
- `*_trace_summary.json`
- `*_trace_REPORT.md`

Общий отчёт:

- `audits/_inbox/tz117_error_traces_20260615_232756/TZ117_TRACE_REPORT.md`
- `audits/_inbox/tz117_error_traces_20260615_232756/tz117_trace_summary.json`

D rerun с явным rationale:

`audits/_inbox/tz117_d_rationale_rerun_20260615_231439/`

## Счётчики

| Блок | rows | model_fix | model_break | both_wrong | high_conf_wrong | gold_present | gold_absent | gold_unclear | avg_conf_ok | avg_conf_error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C | 100 | 43 | 8 | 20 | 11 | 100 | 0 | 0 | 0.833333 | 0.738929 |
| D | 924 | 387 | 28 | 31 | 52 | 924 | 0 | 0 | 0.917931 | 0.904746 |
| A | 24 | 0 | 0 | 0 | 0 | 0 | 24 | 0 | null | null |
| B | 25 | 13 | 6 | 0 | 6 | 19 | 0 | 6 | 1.0 | 1.0 |
| E | 20 | 8 | 10 | 0 | 10 | 18 | 0 | 2 | 1.0 | 1.0 |

Примечания:

- A пока без gold-эталона, поэтому матрица строится как `rule->model`, без claims про точность.
- B/E используют ручной gold-срез. `unclear` не считается точностью и вынесен отдельно.
- D первый прогон после обрыва сети дал 7 fallback, поэтому был повторён. Финальный D-прогон: `codex_cli=23/23`, `codex_rationale=23/23`.

## Источники rationale

- C: `reasoning` из существующего Codex JSONL, тот же вызов классификации.
- D: новое поле `rationale` из того же Codex-вызова роли; fallback на `notes` оставлен для старых артефактов.
- A: `close_reason_summary` + первый `evidence_signal` из того же Codex-анализа, без второго вызова.
- B/E: детерминированная trace-причина из ручного review-файла; модельных вызовов нет.

## Безопасность

- Primary/writeback не включались.
- AMO/Tallanto/CRM не изменялись.
- ASR не запускался.
- `stable_runtime` не изменялся.
- Прямые API-ключи OpenAI не использовались; D rerun шёл через Codex CLI с удалёнными `OPENAI_API_KEY/OPENAI_ORG_ID/OPENAI_PROJECT`.
- Raw trace лежит в ignored `audits/_inbox/`; в git идут только код, тесты и этот отчёт.

## Проверки

- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_tz116_offline_modes.py`
  - `17 passed, 1 warning`
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q`
  - `3288 passed, 5 skipped, 1 warning`

## llm_calls_total

- Новые вызовы Codex для TZ-117: `23` в D rerun.
- Генератор trace-файлов новых модельных вызовов не делает: `calls_model=false`.

## Остаточные риски

- A требует отдельного gold-набора, иначе можно анализировать только расхождения эвристика ↔ модель.
- B/E gold-срез малый; метрики нельзя переносить на весь массив без расширения разметки.
- D показывает сильный прирост по ролям на low-confidence зоне, но high-confidence wrong остаётся заметным: `52` реплики с `conf>=0.8`.
