# P0 model-led complaint, 2026-06-22

Ветка: `codex/p0-model-led-complaint`

База: `main @ 8ffb752`

ТЗ: `/Users/dmitrijfabarisov/Claude Projects/Foton/2026-06-21_TZ_P0_vedet_model_determinizm_verificiruet.md`

## Read-only корень

Проверка на исходном поведении:

```text
detect_high_risk_input_markers("А тестирование нужно? Ребёнок в 6 классе, я просто не понимаю, нас уже в группу или сначала тест?")
=> ('complaint',)
```

После правки:

```text
OFF => ('complaint',)
ON  => ()
TELEGRAM_P0_MODEL_LED in pilot_gold_v1 => False
```

## Что изменено

- Добавлен флаг `TELEGRAM_P0_MODEL_LED`, default OFF, в `pilot_gold_v1` не добавлен.
- `TELEGRAM_P0_MODEL_LED=1` включает JSON-поля model-P0 на direct-path, чтобы clean-профиль тоже мог получить `is_p0/p0_kind`.
- Размытый `complaint` фильтруется только в direct-path обвязке и только под новым флагом.
- `refund`, `legal`, `payment_dispute` и общий `p0_recall_spec.codes_from_text/hard_codes_from_text` не изменены.
- Complaint-only больше не пре-блокирует direct-path, если нет narrow backstop.
- Narrow deterministic backstop оставлен для явных жалоб/child-safety: `жалоба`, `безобразие`, `накричали на ребёнка`, `ребёнок один остался`, `никто не подошёл`, `напишу везде какие вы`, близкие child-safety формулировки.
- Direct prompt получил инструкцию отличать реальную жалобу от растерянности.
- Поздние слои (`apply_high_risk_content_guards`, `apply_autonomy_matrix_guard`, authoritative gate, deal action final P0, humanity guard) не могут заново поднять отфильтрованный fuzzy complaint; реальные hard-маркеры остаются блокирующими.

## Тесты

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_subscription_llm_draft_provider.py -k "p0_model_led or direct_path_model_p0 or direct_path_p0_complaint or child_safety_complaint or child_incident_complaint or benign_teacher"
=> 20 passed, 474 deselected
```

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests
=> 3494 passed, 5 skipped, 1 warning
```

## Симулятор

Набор: `product_data/telegram_dynamic_test_sets/p0_model_led_micro_20260622.jsonl`

Снапшот: `product_data/knowledge_base/kb_release_20260612_v6_7_staging_r4_1/kb_release_v3_snapshot.json`

Судья: `v9.1`

Провайдер: `gpt-5.5`, `bot-mode=codex`, `client-mode=codex`, `judge-mode=codex`, `parallel=4`

Временный `CODEX_HOME`: `/tmp/mango_codex_home_fast_p0_model_led`, только для `service_tier="fast"`; основной `~/.codex/config.toml` не менялся.

Итоговые папки:

| Профиль | Флаг | Папка | dialogs | turns | FAIL | hard | PASS | PWN | config_invalid |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| clean | OFF | `runs/20260622_p0_model_led_clean_OFF_codexclient` | 11 | 25 | 2 | 2 | 3 | 6 | false |
| clean | ON | `runs/20260622_p0_model_led_clean_ON_codexclient_v3` | 11 | 25 | 0 | 0 | 3 | 8 | false |
| pilot_gold_v1 | OFF | `runs/20260622_p0_model_led_pilot_OFF_codexclient` | 11 | 24 | 1 | 1 | 7 | 3 | false |
| pilot_gold_v1 | ON | `runs/20260622_p0_model_led_pilot_ON_codexclient_v3` | 11 | 23 | 0 | 0 | 5 | 6 | false |

Сырьевой контроль по транскриптам:

| Профиль | Флаг | POS manager/complaint | NEG non-manager | complaint_flags | preblocked |
|---|---:|---:|---:|---:|---:|
| clean | OFF | 2 | 0 | 10 | 13 |
| clean | ON | 1 | 0 | 3 | 6 |
| pilot_gold_v1 | OFF | 2 | 0 | 11 | 14 |
| pilot_gold_v1 | ON | 0 | 0 | 3 | 6 |

Итого для живого профиля: ложный complaint на POS снят, реальные NEG не протекли (`NEG non-manager = 0`), дельта hard-fail ON-OFF = `-1`, новых hard-fail = `0`.

## Примечания

- Первые папки `runs/20260622_p0_model_led_*_retry` с `client-mode=fake` не использовать как смысловой замер: fake-client игнорирует конкретные `behaviors` и подставляет стандартные реплики. Они оставлены только как инфраструктурный след.
- Вердикт по качеству/безопасности не выношу; это сырьё для регрейда Claude #1.

## ACK

- `codes_from_text`, `hard_codes_from_text`, `REFUND/LEGAL/PAYMENT_DISPUTE` не редактировались.
- `pilot_gold_v1` не менялся.
- `no_auto_send` / `manager_approval_required` не ослаблялись.
- Live/AMO/Tallanto/stable_runtime не трогались.
