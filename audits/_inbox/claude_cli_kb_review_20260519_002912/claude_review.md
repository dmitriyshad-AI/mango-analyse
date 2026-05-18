# Claude CLI KB Review

Artifact: product_data/knowledge_base/kb_release_20260518_v3_3_bot_pack
Date: 2026-05-19

Verdict: PASS_WITH_NOTES

## What Passed

### Формальная готовность
- Тип пакета: **bot pack** (`schema_version: kb_distribution_packs_v1`, `package_type: bot`).
- Полный набор обязательных файлов: `manifest.json`, `quality_report.json`, `semantic_review.json`, `SEMANTIC_REVIEW.md`, `BOT_USAGE_CONTRACT.md`, `ACTIVE_BRAND_RULES.md`, `README_FOR_BOT.md`, `brand_rules.yaml`, `bot_policy.yaml`, `client_safe_facts_foton.jsonl`, `client_safe_facts_unpk.jsonl`, `manager_only_or_internal_facts.jsonl`, `facts_registry.{jsonl,csv,yaml}`, `bot_template_registry.json`, `bot_fact_index.json`, `post_filter_registry.json`, `source_registry.json`.
- `manifest.json`: `formal_pass=true`, `semantic_pass=true`, `semantic_blocking_findings=0`, `safety.client_auto_send=false`, `safety.active_brand_required=true`, `safety.crm_write=false`, `safety.tallanto_write=false`.
- `quality_report.json`: все 13 чеков passed, `control_numbers.missing=[]` (25 контрольных чисел совпали), `quality_passed=true`.
- Безопасный gate перезапущен независимо на v3.3 handoff:
  `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/run_kb_semantic_review.py --release-dir product_data/knowledge_base/kb_release_20260518_v3_3_handoff_for_claude_and_team`
  → `formal_quality_passed=true`, `semantic_pass=true`, `blocking_findings=0`, `findings=[]`.

### Бренд-изоляция (главное правило проекта)
- Прогон по двум client-safe JSONL (664 факта суммарно): **0 утечек** запрещённых терминов в `client_safe_text`/`fact_text` (УНПК/АНО ДПО/НОУ УНПК/kmipt.ru/@unpk_mipt/телефоны УНПК в Фотон-файле; Фотон/ЦДПО/ЦРДО/cdpofoton/Т-Банк/Долями/телефоны Фотон в УНПК-файле).
- 0 утечек юр.номеров и КНД 1151158 в `allowed_for_client_answer=true` (проверены: 70369, Л035-01255-50/01195871, 0000547, 77753, Л035-01255-50/02496431, 1151158).
- 0 утечек debug-тегов в client_safe_text (`fact_id`, `source_id`, `freshness`, `JSON`, `AMO`, `Tallanto`, `CRM`).
- `brand_rules.yaml`: подтверждённая формула `client_default_relationship_answer` совпадает с CLAUDE.md и `bot_policy.yaml::theme_routes.brand_relationship`.
- `ACTIVE_BRAND_RULES.md`: явные allowlist/blocklist для обоих ботов, без cross-brand упоминаний в инструкции для боте.
- Smoke (Stage 6, реальный Codex, snapshot v3.3): FOTON 20 строк, UNPK 20 строк, `brand_separation_violation=0` оба бренда, `unsupported_numeric_promises=0`, `high_risk_route_relaxed=0`, `baseline_manager_only_relaxed=0`, `errors=0`.

### Manager-only слой
- 304 записи в `manager_only_or_internal_facts.jsonl`. Проверено выборкой: `allowed_for_client_answer=false`, маршруты `manager_only`/`manager_handoff_only`/`draft_for_manager`, у `manager_check_text` явные `client_blocked: ...` теги. Ни один manager-only факт не имеет `allowed_for_client_answer=true`.
- Чувствительные темы боту по контракту запрещены к клиенту: рассрочка 6/12 мес. через менеджера, индивидуальные занятия с неподтверждённым прайсом, кросс-бренд контакты — все ушли в manager-only.

### Шаблоны и контракт
- `BOT_USAGE_CONTRACT.md` item 12: запрет дословной подстановки `client_safe_text` остаётся.
- `BOT_USAGE_CONTRACT.md` item 13–14: введён `bot_template_required` и обязательная запись в `bot_template_registry.json`.
- В реестре: **115 фактов** с `bot_template_required=true`, **115 записей** в `bot_template_registry.json` (`templates_total=115`). 0 потерянных и 0 лишних: `set(required) == set(templates by fact_id)`.
- Все 115 шаблонов имеют `fallback_route="manager_only"` и текст инструкции, явно запрещающий копирование client_safe_text. Это закрывает v3.2 P2 #1 («Фотон: материнский капитал — 3.»).

### Свежесть фактов
- v3.3 закрыл главный P2 v3.2: **664/664 факта имеют одновременно `freshness_check_date` и `valid_until`** (в v3.2 было 241 факт без `valid_until`).

### Post-filter
- `post_filter_registry.json` явно различает `matcher_fields = ["phrases","regex_patterns"]` и `human_only_fields = ["pattern_descriptions"]`. `pattern_descriptions_total=2` (описательные строки изолированы и не уйдут в матчер). Закрывает v3.2 P2 #4.
- 88 запрещённых фраз и 3 regex (промокоды/гарантии/«оплатите сейчас») — покрывают brand-leaks, скрытие природы бота, debug-теги, юр.номера, кросс-бренд скрипты, обещания результата.

### Гигиена пакета
- В bot pack v3.3 **нет** `approval_queue_for_rop_v3.csv` и full snapshot — это закрывает v3.2 P2 #5 (лишние артефакты не уходят в runtime бота). Approval-queue видно только в handoff-пакете.

## Blocking Issues

Нет.

## Non-Blocking Risks

1. **9 ценовых фактов с конкретными ₽ помечены `bot_template_required=false`**, тогда как 58 ценовых требуют шаблона. Примеры:
   - `fact:v3:foton:prices_regular_2026_27_offline_5_11_class_after_2026_07_01_four_weeks` — «...4 недели — 11 400.»
   - `fact:v3:unpk:prices_regular_2026_27_patsayeva_2x_week_year` — «...год — 107 820.»
   - `fact:v3:foton:prices_regular_2026_27_offline_5_11_class_after_2026_07_01_four_weeks_new` — «...для новых учеников — 8 900.»
   Контракт item 12 в любом случае запрещает дословную подстановку, а smoke не нашёл `unsupported_numeric_promises`. Тем не менее логически непоследовательно: либо `bot_template_required=true` для всех ценовых фактов, либо явное правило, почему эти 9 — исключения.

2. **37 discount-фактов разрешены клиенту без `bot_template_required`**. Большинство содержит условие прямо в `client_safe_text` («для семей, уже оплативших 2026/27 год», «для многодетных семей», «для сотрудников МФТИ», «после завершения смены другом») — это уже улучшение по сравнению с v3.2. Но два примера остаются «машинными»:
   - `fact:v3:foton:discounts_multichild_rule` — «Фотон: скидка, правило — 10% многодетным. Применяется как любая другая — не суммируется с другими скидками..» (двойная точка, шероховатая формулировка).
   - `fact:v3:unpk:discounts_multichild_rule` — то же.
   В клиента такой текст уйти не должен (item 12), но лучше либо включить `bot_template_required=true`, либо переписать в человеческой форме.

3. **«smoke50» по факту 20+20=40 строк**. Папка названа `kb_release_20260518_v3_3_smoke50_codex`, но `rows_total=20` в каждом бренде. Это не меняет результат (0 нарушений), но имя вводит в заблуждение и не даёт сравнить покрытие с v3.2 (где была реальная 50). Стоит либо вернуть выборку 50 на бренд, либо переименовать.

4. **Внутренний `manager_check_text` иногда содержит «черновую» формулировку с тегами**: «Фотон: контакты — @unpkmfti. [client_blocked: cross_brand_text, other_brand_term:@unpkmfti, ...]». Это manager-only, но если этот текст когда-нибудь покажут менеджеру в UI без фильтрации тегов — он увидит debug-разметку. Не блокирует, но менеджерский слой UI должен явно отрезать `[client_blocked: ...]` перед показом.

5. Подтверждённые smoke-метрики хороши, но реалистично — это всё ещё 40 вопросов на провайдере. Перед пилотом на лояльных клиентах нужен повторный прогон на тех же 50 диалогах v3.2 + 5–10 свежих кросс-бренд провокаций.

## Missing Checks

- Нет автоматической проверки, что каждый ценовой/скидочный факт с числом имеет либо `bot_template_required=true`, либо записанное в формате «условие + размер» правило. Сейчас это проверяется только вручную.
- Нет gate, проверяющего совпадение `phrases` из `post_filter_registry` с фактической работой пост-фильтра в коде бота (только наличие массива). При желании — отдельный smoke с провокациями.
- Нет автоматической проверки, что `manager_check_text` после фильтра тегов `[client_blocked: ...]` остаётся читаемым менеджеру. Сейчас полагаемся на UI слой.

## Required Regression Tests Or Gates

1. Тест: для каждого факта с `fact_type ∈ {price, discount}` и `allowed_for_client_answer=true` либо `bot_template_required=true`, либо `client_safe_text` содержит явное условие применения и не содержит висящей точки/синтаксических артефактов.
2. Тест: `set(facts where bot_template_required=true) == set(bot_template_registry.templates.fact_id)` (сейчас 115/115; зафиксировать как fail-fast).
3. Тест: 0 пересечений `client_safe_facts_foton.jsonl::client_safe_text` с blocked-terms `when_active_brand_is_foton`; зеркально для УНПК.
4. Smoke50 на бренд (а не 20) — закрепить размер выборки и привязать к имени папки, чтобы `smoke50` всегда означал ≥50 строк.

## Recommended Next Step

- Безопасно использовать v3.3 bot pack для **внутреннего пилота на сотрудниках** в режиме черновиков. Это улучшение к v3.2 по 4 из 5 P2: закрыт `valid_until`-долг, введён `bot_template_registry`, post-filter явно разделён на матчеры/описания, approval-queue убран из bot pack.
- До пилота на лояльных клиентах:
  1. Решить про 9 ценовых + ~2 «машинных» discount-фактов (item 1–2 в рисках) — поставить `bot_template_required=true` или переписать формулировки.
  2. Расширить smoke до настоящих 50 строк на бренд и пересобрать `smoke50_codex`.
  3. В UI менеджера явно отрезать `[client_blocked: ...]` теги перед показом.

## Summary
- formal_pass: да
- semantic_pass: да
- pilot_ready: да с условиями (внутренний пилот на сотрудниках — да; пилот на клиентах — после правки 3 пунктов выше)
- production_ready: нет
