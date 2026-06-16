# TZ-123 после TZ-124: остаток для вопроса вместо ухода

Дата: 2026-06-16  
Ветка: `codex/tz123-tz124-remeasure`  
База ветки: `codex/tz124-slot-anchor` (`b7c2e67`) + точечно перенесён код TZ-123.

## Что проверялось

Задача: после TZ-124 пересобрать набор TZ-123 из остатка, где быстрый параметр реально не назван, и проверить, появляется ли реальное срабатывание вопросного слоя без допроса/циклов, P0 и бренд не ломаются.

Флаги replay:

- `TELEGRAM_DIRECT_PATH_PILOT_CONFIG=pilot_gold_v1`
- `TELEGRAM_ANCHORED_BARE_GRADE=1`
- `TELEGRAM_MEMORY_PROVENANCE=1`
- `TELEGRAM_QUESTION_INSTEAD_OF_HANDOFF=0/1`

Записей в AMO/Tallanto/CRM не было. Сообщения клиентам не отправлялись. ASR не запускался. `stable_runtime` не трогался.

## Метод

Добавлен воспроизводимый измеритель:

- `scripts/run_tz123_tz124_remainder_measure.py`

Он делает три шага:

1. Сканирует исторические `runs/**/dynamic_dialog_transcripts.jsonl` и берёт только сохранённые C0-уходы: `draft_for_manager + action_decision.answer_only`.
2. Применяет текущий TZ123 layer поверх сохранённого результата, но уже с TZ124-парсером слотов (`TELEGRAM_ANCHORED_BARE_GRADE=1`). Так отсекаются случаи типа `физика 8 онлайн`, где класс теперь распознан.
3. Делает полный текущий replay с `QUESTION=OFF`; из него строит актуальный остаток, где бот сейчас действительно ушёл бы к менеджеру. Только этот остаток должен идти в `QUESTION=ON`.

## Результат

Layer-level scan по сохранённым C0:

| Метрика | Значение |
|---|---:|
| Исторических target-кандидатов после TZ124-фильтра | 17 |
| `grade` | 11 |
| `format` | 1 |
| `subject` | 2 |
| `time` | 3 |

Текущий replay `QUESTION=OFF` по этим 17 кандидатам:

| Метрика | Значение |
|---|---:|
| Диалогов | 17 |
| Ходов | 17 |
| `draft_for_manager + answer_only` после текущего replay | 0 |
| `bot_answer_self_for_pilot + answer_only` | 17 |
| hard gate failures | 0 |
| violated_gates | `{}` |
| verdict | `PASS_WITH_NOTES` × 17 |
| llm_calls_total | 51 |

Итог: актуальный остаток после TZ124 и свежего replay пустой (`remainder_candidates=0`). Поэтому `QUESTION=ON` на актуальном остатке не запускался: нет случаев, где текущий бот действительно ушёл бы к менеджеру из-за неназванного быстрого параметра.

## Вывод

`fired>0` появился на уровне сохранённых исторических C0-уходов: текущий TZ123 layer видит 17 настоящих кандидатов после TZ124-фильтра.

Но в полном текущем replay эти же 17 случаев уже не уходят в `draft_for_manager`: базовый direct path отвечает сам (`bot_answer_self_for_pilot`) и поэтому gate-in TZ123 не достигается. Для текущего живого пути фактический dynamic `fired` остаётся `0`, потому что актуального остатка нет.

Это не ошибка TZ123: слой включается только на `draft_for_manager + answer_only`. Сейчас после TZ124 и текущего direct path такие случаи в измеренном остатке исчезли.

## Semantic review

Verdict: `PASS_WITH_NOTES`.

Что прошло:

- P0 не перехватывался: `hard_gate_failures=0`, `violated_gates={}`.
- Бренд не смешивался в проверенном replay.
- Допроса/циклов нет: replay одноходовый, актуальный остаток пустой.
- Сырые тексты и replay-артефакты лежат в ignored `audits/_inbox`, не в git.

Неблокирующий риск:

- На сохранённых исторических C0 слой имеет 17 потенциальных срабатываний, но это не доказывает прирост на текущем живом пути, потому что свежий replay уже не даёт target-route.

Manual gate для регрейда:

- Если Claude найдёт актуальный `draft_for_manager + answer_only` с неназванным быстрым параметром, надо добавить его в остаток и повторить ON replay.
- Любой вопрос на P0/жалобе или по чужому бренду = BLOCKED.

## Артефакты

- Summary: `audits/_inbox/tz123_tz124_remainder_20260616/summary.json`
- Initial layer candidates: `audits/_inbox/tz123_tz124_remainder_20260616/initial_layer_candidates.jsonl`
- Current OFF replay: `audits/_inbox/tz123_tz124_remainder_20260616/off_replay/`
- Empty remainder: `audits/_inbox/tz123_tz124_remainder_20260616/remainder_rows.jsonl`

## Проверки

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q -k 'tz123_question or question_instead_of_handoff' tests/test_subscription_llm_draft_provider.py tests/test_telegram_dynamic_client_sim.py
=> 13 passed, 584 deselected

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_tz124_slot_anchor.py
=> 12 passed

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m py_compile scripts/run_tz123_tz124_remainder_measure.py
=> ok
```
