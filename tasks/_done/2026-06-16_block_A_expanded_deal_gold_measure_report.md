# Block A: расширенный gold-набор по сделкам

Дата: 2026-06-16

## Что проверяли

Цель: добрать gold-набор по закрытым сделкам больше 24 строк, покрыть оба бренда и сравнить модельный разбор с текущей эвристикой по трём бизнес-полям:

- вердикт закрытия сделки;
- риск преждевременного закрытия;
- следующий шаг.

Live-записи и live-чтения AMO/Tallanto/CRM не выполнялись. Использовались только локальные артефакты.

## Источники

- 24 старых precomputed результата:
  `/Users/dmitrijfabarisov/Projects/Mango_tz116_offline/audits/_inbox/tz116_crm_llm_shadow_fixed24_codex_20260615_195654/crm_llm_offline_measure_results.jsonl`
- локальный read-only snapshot:
  `/Users/dmitrijfabarisov/Projects/Mango analyse/stable_runtime/deal_aware_amo_live_snapshot_20260513_v2`
- новые результаты и trace:
  `/Users/dmitrijfabarisov/Projects/Mango_blockA_gold/audits/_inbox/block_a_deal_gold_expanded_20260616/`

## Как добрали набор

Добавлено 8 закрытых сделок из локального snapshot:

- 4 по бренду Foton;
- 4 по бренду UNPK;
- причины закрытия: `Недозвон`, `Архив  (нет связи)`, `Не актуально`, `Действующий клиент`.

Итоговый набор:

- всего строк: 32;
- Foton: 16;
- UNPK: 16;
- новых модельных вызовов через Codex CLI: 8;
- Tallanto lookup: выключен;
- writeback: запрещён.

## Результат сравнения

| Метрика | Эвристика | Модель |
|---|---:|---:|
| Точных совпадений с gold | 23 / 32 | 18 / 32 |
| Доля точных совпадений | 71.875% | 56.25% |
| Разница модели к эвристике |  | -5 |

Типы ошибок:

- обе системы верно: 18;
- обе системы ошиблись: 9;
- модель ухудшила ответ относительно эвристики: 5.

У модели осталось 8 уверенных ошибок, поэтому переводить Block A в основной режим нельзя.

## Смысловой вывод

Вердикт: `PASS_WITH_NOTES`.

Что прошло:

- набор расширен до 32 строк;
- оба бренда покрыты симметрично;
- результат можно использовать для регрейда Claude и Дмитрия;
- live-записи и live-чтения не выполнялись.

Что не прошло:

- модель не стала явно лучше эвристики;
- уверенные ошибки модели остаются;
- ручные gold-метки являются консервативной бизнес-разметкой, а не разрешением на CRM-запись.

Решение: Block A остаётся в режиме тени. Кандидатом на основной режим он станет только после отдельного улучшения и повторного замера без уверенных ошибок.

## Безопасность

- AMO write: нет;
- Tallanto write: нет;
- CRM write: нет;
- live CRM read: нет;
- использован локальный snapshot;
- сырые артефакты лежат в ignored `audits/_inbox/...` и не должны попадать в коммит.

## Что отдавать на регрейд

Для регрейда открыть:

- `/Users/dmitrijfabarisov/Projects/Mango_blockA_gold/tasks/_done/2026-06-16_block_A_expanded_deal_gold_measure_report.md`
- `/Users/dmitrijfabarisov/Projects/Mango_blockA_gold/audits/_inbox/block_a_deal_gold_expanded_20260616/summary.json`
- `/Users/dmitrijfabarisov/Projects/Mango_blockA_gold/audits/_inbox/block_a_deal_gold_expanded_20260616/deal_a_gold_trace_manual.csv`
- `/Users/dmitrijfabarisov/Projects/Mango_blockA_gold/audits/_inbox/block_a_deal_gold_expanded_20260616/new_cases_redacted.csv`

## Остаточный риск

Новые 8 строк взяты из snapshot от 2026-05-13, а не из свежего live AMO. Для решения `shadow vs primary` этого достаточно, но для бизнес-утверждения о текущей CRM-картине нужен отдельный свежий read-only snapshot.
