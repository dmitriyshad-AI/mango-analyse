# Bot Safety Frozen Corpus Report

Дата: 2026-05-10

## Что сделано

После внешнего Claude-аудита controlled allowlist перешли от ручного whack-a-mole к release-gate подходу:

- создан независимый `bot_safety_detector`, который не импортирует regex-правила sanitizer;
- создан builder/validator frozen adversarial corpus;
- frozen corpus собирается из трех слоев: synthetic, hand-curated Claude/GPT findings, random real allowlist rows;
- добавлены ASR-tolerance кейсы: искажения фамилий, адресов, брендов, платежных провайдеров;
- threat model дополнен классом `over_sanitization_cluster_repeat`;
- Stage 15 пересобран на актуальном sanitizer как `v11_frozen_gate`.

## Итоговый Frozen Corpus

Путь:

- `stable_runtime/bot_safety_frozen_corpus_20260510_v3_frozen_gate/bot_safety_adversarial_cases.jsonl`

Размер:

```text
rows: 1312
synthetic: 1100
real_allowlist_random_seed_42: 200
hand_curated_audit: 12
```

Разбиение по основным классам:

```text
money: 350
orphan_surname: 280
real_row_regression: 200
deadline: 147
location: 126
single_name: 70
payment_terms: 35
personal_name: 35
promise: 28
asr_tolerance: 24
```

## Validation Result

Путь:

- `stable_runtime/bot_safety_frozen_corpus_validation_20260510_v4_frozen_gate/summary.json`

Результат:

```text
passed: true
rows: 1312
failures: 0
risk_counts: {}
by_pass_count: {"1": 1312}
```

Это означает, что sanitizer + independent detector проходят frozen release corpus без P0/P1/P2 утечек и без idempotence/fixpoint failures.

## Stage 15 After Rebuild

Путь:

- `stable_runtime/transcript_quality_stage15_export_gate_20260510_v11_frozen_gate/summary.json`

Результат:

```text
passed: true
bot_export_allowlist_rows: 473
blocked_bot_export_rows: 0
stage14_residual_risk_rows: 0
stage14_over_sanitization_rows: 250
```

Bot export risk-counts:

```text
brand: 0
money_or_terms: 0
personal_data: 0
spoken_money_or_terms: 0
messenger_handle: 0
unsafe_placeholder: 0
brand_variant: 0
likely_single_name: 0
fixpoint_not_reached: 0
missing: 0
```

## Кодовые изменения

- `src/mango_mvp/quality/bot_safety_detector.py`
- `src/mango_mvp/quality/bot_safety_frozen_corpus.py`
- `scripts/build_bot_safety_frozen_corpus.py`
- `tests/test_bot_safety_detector.py`
- `tests/test_bot_safety_frozen_corpus.py`
- `src/mango_mvp/insights/sanitizers.py` получил ASR-tolerance hardening для `ФТИ`, `Альфа/Алфа`.

## Ограничения

- Frozen corpus является release-gate для зафиксированного набора классов, а не доказательством абсолютной безопасности всех возможных будущих формулировок.
- `over_sanitization_candidates=250` остается отдельной очередью полезности: это не safety failure, но автономному боту нельзя отдавать такие строки без ROP-review/схлопывания повторов.
- Detector сейчас независимый, но эвристический. NER пока не подключен; это осознанно, потому что текущий gate прошел без него.

## Вывод

Текущий controlled manager-assist allowlist можно считать готовым по safety-критерию. Для автономного Telegram-бота следующий блокер не safety, а полезность и продуктовая политика: нужно разобрать over-sanitization queue и утвердить, какие ответы бот имеет право давать без менеджера.
