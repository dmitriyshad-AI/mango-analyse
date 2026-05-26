# Патч CLAUDE.md: v6.3 каноничен, v3.3 устарела. 2026-05-26 (предложение, применяет Codex/Дмитрий)

Точечная правка секции «Текущая база знаний» (строки 101-142). Заменить блок целиком на новый.
Обоснование: код (бот v2 `dialogue_contract_pipeline.py:36` + симулятор) использует v6.3, а CLAUDE.md
называл текущей v3.3 — рассинхрон документации с фактом.

## БЫЛО (строки 101-142)
```
## Текущая база знаний v3.3

Актуальный машинный релиз:

```text
product_data/knowledge_base/kb_release_20260518_v3_3/
```
… (handoff/employee/bot/smoke20 пакеты v3.3) …

Статус v3.3:
- `formal_pass=true`;
- `semantic_pass=true`;
- … все 664 факта имеют valid_until …
```

## СТАЛО (новый блок)
```markdown
## Текущая база знаний v6.3 (каноничная)

Каноничный машинный релиз (его использует код бота и симулятор):

```text
product_data/knowledge_base/kb_release_20260520_v6_3_team_answers/
```

Снимок для бота/симулятора: `kb_release_20260520_v6_3_team_answers/kb_release_v3_snapshot.json`
(захардкожен в `src/mango_mvp/channels/dialogue_contract_pipeline.py` как `DEFAULT_KB_SNAPSHOT_PATH`).

Важно про имена версий (три РАЗНЫХ оси, не путать):
- релиз = **v6.3** (`kb_release_20260520_v6_3_team_answers`);
- версия СХЕМЫ снимка = **v3** (поэтому файл называется `kb_release_v3_snapshot.json`);
- версия СБОРЩИКА = **v6.1** (`kb_release_v6_1_builder`, скрипт `build_kb_release_v6_1_team_answers.py`).

Пакеты v6.3: `..._team_answers_bot_pack/`, `..._team_answers_employee_pack/`.

Сборка: ТОЛЬКО через `scripts/build_kb_release_v6_1_team_answers.py` (применяет `release_manifest.yaml`).
НЕ использовать старый `build_kb_release_v3_from_claude_handoff.py`. См. `docs/KB_BUILD_RUNBOOK_2026-05-26.md`.

Статус v6.3 (по актуальному `quality_report.json`/`semantic_review.json`):
- `quality_passed=true`, `semantic_pass=true`, blocking findings: 0;
- гейты целостности зелёные: `text_number_grounded`, `field_ranges_ok`, `weekly_frequency_is_plausible`, `control_numbers_present`;
- 838 фактов, client-safe 473 (на пересборку 2026-05-27; число берётся из quality_report, не хардкодить);
- режим: внутренний пилот на сотрудниках и лояльной подготовленной группе клиентов только как черновики
  с обязательным одобрением менеджера; публичный трафик и автоотправка не разрешены.

### Устаревшее (НЕ источник правды)
- **v3.3** (`kb_release_20260518_v3_3*`) — устарела, оставлена для истории. Не использовать как текущую.
```

## Также проверить (отдельно, не часть этого блока)
- В секции «Какой пакет проверять» (ниже по CLAUDE.md) пути v3.3 заменить на v6.3-аналоги
  (`kb_release_20260520_v6_3_team_answers_bot_pack/` и т.д.) — иначе аудит укажет на устаревший пакет.
- Подтвердить, какой снимок читает ПРОД-рантайм бота (в `stable_runtime/deploy/` захардкоженного пути
  нет → берётся дефолт кода v6.3 или из конфига запуска). Если конфиг — синхронизировать на v6.3.
