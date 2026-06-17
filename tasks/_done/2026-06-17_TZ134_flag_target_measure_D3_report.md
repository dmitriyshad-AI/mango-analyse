# TZ-134 flag target OFF/ON measure report

Дата: 2026-06-17
Ветка: `codex/measure-flags-honest`
Ревизия кода: `ae957b6`

## Что сделано

1. Собран runnable-набор:
   - `product_data/telegram_dynamic_test_sets/flag_target_set_v1.jsonl`
   - 33 строки: `simulator_spec`, `judge_spec`, 31 persona.
   - База фактов: `product_data/knowledge_base/kb_release_20260612_v6_7_staging_r4_1/kb_release_v3_snapshot.json`.
   - `confirmed_facts_snapshot`: 120 клиентских безопасных фактов Фотона + 120 клиентских безопасных фактов УНПК из `v6_7_staging_r4_1`.

2. Проверены обязательные POS-факты в `client_safe_facts`:
   - цена Фотон онлайн 5-11: 29 750 ₽ за семестр, 47 250 ₽ за год;
   - цена УНПК очно 5-11: 49 000 ₽ за семестр, 82 000 ₽ за год;
   - адрес Фотона: Верхняя Красносельская;
   - старт учебного года УНПК;
   - скидка на второй предмет онлайн: Фотон 30%, УНПК 20%;
   - городской лагерь Фотон + цена;
   - скидка многодетным 10%;
   - онлайн-формат Фотона + записи.

3. Важная пометка по фактам:
   - `ft_t122_format_01`: актуальная формулировка в `v6_7_staging_r4_1` — новые онлайн-занятия проходят на SohoLMS; курсы, начавшиеся до лета 2026, завершаются в MTS-Link; записи сохраняются. Старое ожидание "только MTS-Link" не считать отдельным ложным FAIL.

## Прогоны

Общее:
`gpt-5.5`, judge prompt `v9`, `--parallel 4`, `pilot_gold_v1`, `memory codex low`, `semantic codex medium`, `allow_default_autonomy` не менялся отдельно.

Из-за локальной ошибки глобального Codex config (`service_tier=default`, текущий CLI принимает только `fast`/`flex`) использован временный `CODEX_HOME` в `/private/tmp` с тем же логином и `service_tier="fast"`. Глобальный `~/.codex` не менялся.

### OFF

Флаги TZ-122/TZ-124/TZ-123 не выставлялись.

Папка:
`runs/flag_target_OFF_ae957b6`

Сырьё для регрейда:
`runs/flag_target_OFF_ae957b6/dynamic_dialog_transcripts.jsonl`

Итог:
- dialogs: 31
- turns: 89
- PASS: 14
- PASS_WITH_NOTES: 17
- FAIL: 0
- hard_gate_failures: 0
- infra_error: 0

SHA256 transcript:
`54a4c092a215e12bb541c8b4b89c28485229eb6de3ae6af81b2bd82a232018b9`

### ON

Выставлены ровно три флага:

```bash
TELEGRAM_WRONG_INTENT_FACT_CALIBRATION=1
TELEGRAM_ANCHORED_BARE_GRADE=1
TELEGRAM_QUESTION_INSTEAD_OF_HANDOFF=1
```

Папка:
`runs/flag_target_ON_ae957b6`

Сырьё для регрейда:
`runs/flag_target_ON_ae957b6/dynamic_dialog_transcripts.jsonl`

Итог:
- dialogs: 31
- turns: 90
- PASS: 16
- PASS_WITH_NOTES: 14
- FAIL: 1
- hard_gate_failures: 1
- infra_error: 0

Единственный FAIL:
- `ft_t124_neg_noclass_01`
- violated_gates: `fabrication`
- rationale судьи: бот выдал цены до того, как клиент назвал класс/формат, и далее автономно рекомендовал уровень без подтверждённого факта.

SHA256 transcript:
`586eb4ce5a525c0635e87ab07f5af528792ab116f397c5b00392f7629394b0c9`

## Что отдать Claude #1

1. `runs/flag_target_OFF_ae957b6/dynamic_dialog_transcripts.jsonl`
2. `runs/flag_target_ON_ae957b6/dynamic_dialog_transcripts.jsonl`
3. Этот отчёт.

Критичная точка ручного регрейда:
`ft_t124_neg_noclass_01` в ON, потому что именно он дал новый hard FAIL.
