# TZ-123 C0 surface before enabling

Дата: 2026-06-16
Ветка: `codex/tz123-question-instead-of-handoff`
HEAD: `d64fb958`

## Задача

Проверить реальную поверхность для `TELEGRAM_QUESTION_INSTEAD_OF_HANDOFF` перед включением: пройти по C0-уходам `draft_for_manager + action_decision.answer_only` и отделить случаи, где уход вызван именно нехваткой быстрого клиентского параметра (`grade`, `subject`, `format`, `time`), а не gate-находкой, закрытием, CRM/действием или P0.

## Метод

1. Просканированы доступные реальные транскрипты:
   `/Users/dmitrijfabarisov/Projects/Mango analyse/runs/**/dynamic_dialog_transcripts.jsonl`.
2. C0 определялся как финальный `bot_route == draft_for_manager` + `bot_action_decision.action == answer_only`.
3. Для каждого C0 восстановлен `SubscriptionDraftResult` и применён текущий слой `apply_question_instead_of_handoff_layer` с `TELEGRAM_QUESTION_INSTEAD_OF_HANDOFF=1`.
4. Отдельно собран replay-набор только из target-like случаев, где быстрый параметр мог разблокировать факт:
   - `product_data/telegram_dynamic_test_sets/tz123_c0_surface_real_20260616.jsonl`
   - `product_data/telegram_dynamic_test_sets/tz123_c0_surface_real_replay_20260616.jsonl`

## Surface scan

По всем историческим доступным прогонам найдено 86 C0-ходов.

Разбор текущим слоем:

| Класс | Ходов |
|---|---:|
| `single_missing_slot_question` | 26 |
| `no_unlocking_slot` | 47 |
| `authoritative_gate_not_pass` | 7 |
| `regate:downgrade_keep_text` | 6 |

По слотам среди 26 механических срабатываний:

| Слот | Ходов |
|---|---:|
| `grade` | 20 |
| `time` | 3 |
| `subject` | 2 |
| `format` | 1 |

Критически важное разделение:

- На актуальном/близком финальном профиле найден 1 strict C0: `20260616_main_tz113_114_115_profile_smoke18 / pilot_smoke18_17_unpk_lead_pii_no_echo / turn 1`.
- Этот ход не target для TZ-123: это запись/лид с PII и недостающими CRM-полями, а не нехватка быстрого параметра для ответа на факт-вопрос. Слой корректно пропустил: `no_unlocking_slot`.
- Остальные target-like случаи пришли из экспериментальных `gain/autonomy` и micro-прогонов; часть из них является дублями или ложной поверхностью.

## Target replay

Для OFF→ON выбран минимальный набор из 3 реальных target-like случаев:

1. `autonomy_unpk_real_002` turn 1: клиент спрашивает стоимость и варианты оплаты без класса.
2. `gain_fact_p01_unpk_monthly_price` turn 1: клиент спрашивает помесячную стоимость без параметров.
3. `gain_fact_p04_foton_schedule_days` turn 1: клиент спрашивает дни расписания без класса/предмета/формата.

Команды replay:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src TELEGRAM_DIRECT_PATH_PILOT_CONFIG=pilot_gold_v1 TELEGRAM_QUESTION_INSTEAD_OF_HANDOFF=0 \
python3 scripts/run_telegram_dynamic_client_sim.py \
  --scenarios product_data/telegram_dynamic_test_sets/tz123_c0_surface_real_20260616.jsonl \
  --replay-from product_data/telegram_dynamic_test_sets/tz123_c0_surface_real_replay_20260616.jsonl \
  --out-dir runs/20260616_tz123_c0_surface_real_OFF \
  --parallel 3 --bot-mode codex --judge-mode fake --client-mode fake \
  --memory-mode off --semantic-mode fake --semantic-verifier-mode fake \
  --judge-prompt-version v9.1 --timeout-sec 180
```

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src TELEGRAM_DIRECT_PATH_PILOT_CONFIG=pilot_gold_v1 TELEGRAM_QUESTION_INSTEAD_OF_HANDOFF=1 \
python3 scripts/run_telegram_dynamic_client_sim.py \
  --scenarios product_data/telegram_dynamic_test_sets/tz123_c0_surface_real_20260616.jsonl \
  --replay-from product_data/telegram_dynamic_test_sets/tz123_c0_surface_real_replay_20260616.jsonl \
  --out-dir runs/20260616_tz123_c0_surface_real_ON \
  --parallel 3 --bot-mode codex --judge-mode fake --client-mode fake \
  --memory-mode off --semantic-mode fake --semantic-verifier-mode fake \
  --judge-prompt-version v9.1 --timeout-sec 180
```

Результат:

| Метрика | OFF | ON |
|---|---:|---:|
| dialogs | 3 | 3 |
| turns | 3 | 3 |
| fail | 0 | 0 |
| hard_gate_failures | 0 | 0 |
| `question_instead_of_handoff.fired` | 0 | 0 |
| `question_instead_of_handoff.skipped.route_not_draft_for_manager` | - | 3 |

Почему `fired=0`: на текущем дереве модель/финализация уже дают `bot_answer_self_for_pilot` по всем трём replay-входам. Слой видит финальный route как self, а не `draft_for_manager`, и корректно не вмешивается.

## Transcript check

OFF→ON не дал P0/brand-регрессий:

- `hard_gate_failures=0` в обоих плечах;
- `violated_gates={}` в обоих плечах;
- `config_validity.invalid=false` в обоих плечах.

В ON все 3 хода имеют:

```json
{"enabled": true, "status": "skipped", "reason": "route_not_draft_for_manager"}
```

## Исключённые случаи

Исключены из target surface:

- CRM/lead/identity случаи: действие не `answer_only`, либо недостающие данные не являются быстрым параметром для факт-ответа.
- Gate-находки и re-gate: слой не должен обходить authoritative gate.
- Micro-дубли: полезны для unit-проверки безопасности, но не доказывают реальную поверхность.
- Случаи вида `физика 8 онлайн`, где слой видит missing `grade`: это проблема извлечения/провенанса слота, а не вопрос вместо ухода. Включение TZ-123 здесь рискует создать лишний переспрос при уже названном классе.

## Вывод

Surface для TZ-123 на текущем боевом профиле мизерный:

- strict C0 на актуальном профиле: 1;
- target strict C0 на актуальном профиле: 0;
- target replay из исторических/gain-прогонов: 3;
- фактических превращений ухода в вопрос на текущем дереве: 0.

Рекомендация: держать `TELEGRAM_QUESTION_INSTEAD_OF_HANDOFF` OFF. Сейчас это безопасный, но низкоценный слой: он не ухудшает replay-набор, но и не даёт измеримого выигрыша на доступной реальной поверхности. Для будущего включения нужен новый сигнал, что strict C0 снова появился в боевом профиле, либо отдельная правка извлечения слотов для случаев вроде `физика 8 онлайн`.

## Артефакты

- Surface rows: `/tmp/tz123_c0_surface_rows.jsonl`
- Replay OFF: `runs/20260616_tz123_c0_surface_real_OFF`
- Replay ON: `runs/20260616_tz123_c0_surface_real_ON`
- Target set: `product_data/telegram_dynamic_test_sets/tz123_c0_surface_real_20260616.jsonl`
- Replay source: `product_data/telegram_dynamic_test_sets/tz123_c0_surface_real_replay_20260616.jsonl`

