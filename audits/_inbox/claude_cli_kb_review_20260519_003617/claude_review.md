# Claude CLI KB Review

Artifact: `product_data/knowledge_base/kb_release_20260518_v3_3_bot_pack`
Date: 2026-05-19
Reviewer: Claude CLI (kb-reviewer agent), read-only
Package type: bot pack (machine-readable for Telegram-бота, режим черновиков)

Verdict: **PASS_WITH_NOTES**

## What Passed

### Формальная готовность
- `manifest.json`: `formal_pass=true`, `semantic_pass=true`, `package_type=bot`, `client_safe_facts_total=360`, `facts_total=664`, `semantic_blocking_findings=0`.
- `quality_report.json`: все 13 чек-боксов `true`, `quality_passed=true`, ни одного `blocking_failure`, все 25 контрольных чисел (44600, 74500, 49000, 82000, 29750, 47250, 98000, 75000, 120000, 89900, 83800, 16900, 27720, 18800, 34400, 3900, 6900, 23000, 18900, 94500, 33000, 50000, 11900, 56500, 94000) найдены — `missing: []`.
- `semantic_review.json`: `semantic_pass=true`, `blocking_findings=0`, `findings_total=0`, `formal_quality_passed=true`.
- На месте: `BOT_USAGE_CONTRACT.md`, `ACTIVE_BRAND_RULES.md`, `README_FOR_BOT.md`, `SEMANTIC_REVIEW.md`, `brand_rules.yaml`, `bot_policy.yaml`, `post_filter_registry.json`, `bot_template_registry.json`, `bot_fact_index.json`, `source_registry.json`, `facts_registry.{jsonl,csv,yaml}`, `client_safe_facts_{foton,unpk}.jsonl`, `manager_only_or_internal_facts.jsonl`.

### Что улучшилось по сравнению с v3.2 (релиз `2026-05-18 v3.3`)
1. **`post_filter_registry.json` теперь корректно разделён**: появились явные поля `matcher_fields: [phrases, regex_patterns]` и `human_only_fields: [pattern_descriptions]`, контракт (§15) фиксирует, что `pattern_descriptions` — человеческие пояснения, а матчер использует только `phrases` и `regex_patterns`. Это закрывает риск №4 из ревью v3.2.
2. **`valid_until` теперь проставлен у всех 360 клиентских фактов** (`grep "valid_until": ""` → 0). Это закрывает риск №3 из ревью v3.2 (241 факт без `valid_until`).
3. **Появился `bot_template_registry.json`**: 173 шаблона. Число точно совпадает с количеством фактов с `bot_template_required=true` в client-safe (foton 93 + unpk 80 = 173). Контракт (§13-14) запрещает дословную подстановку, fallback — `manager_only`.
4. **BOT_USAGE_CONTRACT.md (§12) явно запрещает подставлять `client_safe_text` дословно** клиенту или менеджеру: «использует факт как источник смысла и собирает нормальную фразу из утверждённого шаблона, `structured_value.raw_value` и контекста». Это снимает остроту риска №1 из ревью v3.2.
5. **`approval_queue_for_rop_v3.csv` отсутствует в bot pack** (есть только `facts_registry.csv` для удобства просмотра). Замечание №5 v3.2 учтено.

### Бренд-разделение и клиентский слой
- В `client_safe_facts_foton.jsonl` — 0 упоминаний «УНПК».
- В `client_safe_facts_unpk.jsonl` — 0 упоминаний «Фотон».
- 0 утечек `fact_id`, `source_id`, `claude_layer`, `АНО ДПО`, `НОУ УНПК`, `ООО ЦДПО`, `ООО ЦРДО` в `client_safe_text`.
- 0 промокодов в `client_safe_text`; `bot_policy.yaml.promo_codes_policy.bot_handles_promo: false`; шаблон ответа: «передам менеджеру — он подскажет актуальные акции и промокоды».
- Единственное упоминание «лицензия» в client-safe — безопасная фраза «У учебного центра есть лицензия на образовательную деятельность» (по обоим брендам). Номера лицензий и КНД 1151158 в клиентском тексте отсутствуют.
- 0 фактов с `route_policy=manager_only` в client-safe; 0 фактов с `allowed_for_client_answer=true` в `manager_only_or_internal_facts.jsonl` — фильтр корректный.
- В `manager_only_or_internal_facts.jsonl` пример блокировки чистый: `safety_block_reasons=["cross_brand_text", "other_brand_term:@unpkmfti", "unconfirmed_foton_contact"]`, `client_safe_text=""`, `manager_display_text` без служебных пометок (по §16).

### Контракт и режим работы
- `BOT_USAGE_CONTRACT.md` (16 пунктов) полностью соответствует CLAUDE.md: активный бренд из канала, `manager_only` для возвратов/жалоб/угроз/спорной оплаты, не отправлять клиенту до отдельного решения, AMO/Tallanto только read-only, при противоречии AMO↔Tallanto — `manager_only`.
- `bot_policy.yaml.bot_answer_self_rollout`: `enabled_for_staff_tests=true`, `enabled_for_loyal_prepared_clients=true`, `enabled_for_public_traffic=false` — соответствует поэтапному пилоту из CLAUDE.md.
- `bot_policy.yaml` корректно описывает `theme_routes` для refund/complaint/legal_threat (`risk: high, route: manager_only, collect: false, ZERO collect`), brand_relationship (только утверждённая нейтральная фраза), matkap и tax_deduction (info-ответы с маршрутизацией, без раскрытия юр.лица и номеров лицензий).
- `brand_rules.yaml` повторяет `core_rule` слово в слово с CLAUDE.md, `forbidden_client_mentions` покрывает оба бренда, заблокированные термины совпадают с `ACTIVE_BRAND_RULES.md` и `post_filter_registry.phrases`.

### Smoke (Stage 6, codex provider)
- `manifest.smoke`: FOTON 20 строк, UNPK 20 строк, `errors=0`, `brand_separation_violation=0`, `high_risk_route_relaxed=0`, `baseline_manager_only_relaxed=0`, `unsupported_numeric_promises=0`, `invalid_topic_ids=0`, `used_kb_context=20/20`, `became_more_substantive=8`, `empty_clarification_reduced=1`.
- Все safety-флаги Stage 6 жёстко в нуле: `client_send=false`, `live_telegram=false`, `write_crm=false`, `write_tallanto=false`, `write_stable_runtime=false`, `run_asr=false`, `run_resolve_analyze=false`, `manager_approval_required=true`.

## Blocking Issues

Нет.

## Non-Blocking Risks

1. **Машинные обрывочные `client_safe_text` всё ещё присутствуют** (всего ~28 коротких текстов вида «Фотон: рассрочка и оплата — Т-Банк.», «Фотон: рассрочка и оплата — Условия Т-Банка обновлять минимум раз в квартал.», «Фотон: рассрочка и оплата — 1-2 минуты.», «Фотон: материнский капитал — Договор.», «Фотон: онлайн-платформа, онлайн — да.», «УНПК: в интенсив 2026 входит: 3 пробника.»). Все они закрыты `bot_template_required=true` и попадают в `bot_template_registry.json`, fallback — `manager_only`. Контракт §12 в рантайме защитит клиента. Но текст этих фактов всё равно неприятно читается РОПом и сотрудниками в `employee_pack`, и при обновлении контракта легко проскочит регресс. Рекомендация: builder при сборке должен переписать `client_safe_text` короче «факт: значение» — например, «Фотон: рассрочка через Т-Банк (детали уточняет менеджер)». Перенести в тест.
2. **Контактные факты с коротким горизонтом `valid_until=2026-12-31`** (телефоны, телеграм-юзернеймы) — это нормальный консервативный срок, но в декабре 2026 нужно отдельным гейтом обновить все факты темы `contact`/`schedule`. Зафиксировать ответственного владельца в `owner_role`.
3. **Дублирование smoke20 и smoke50**: каталоги `kb_release_20260518_v3_3_smoke20_codex/foton/stage6_kb_enriched_drafts.csv` и `kb_release_20260518_v3_3_smoke50_codex/foton/stage6_kb_enriched_drafts.csv` побайтно идентичны (по 132 строки, `rows_total=20`). Метрики в `stage6_eval_summary.json` тоже одинаковы. Это путает: имя «smoke50» намекает на 50 вопросов, а в манифесте сказано «smoke50: FOTON rows=20, UNPK rows=20». Не блокирует ревью, но либо переименовать каталог, либо реально расширить smoke50 до 50 вопросов.
4. **`quality_report.stage6.status="not_run_by_builder"`** — корректно по архитектуре (smoke запускается отдельным контуром), но в `quality_report` хорошо бы добавить явную ссылку на путь со smoke-результатами для текущего релиза, иначе при последующем аудите легко не найти подтверждение запуска.
5. **«Утренний клуб Предлёнка» в client-safe лагере** (`Фотон: городской летний лагерь, название — Утренний клуб Предлёнка.`) — самостоятельное название продукта, но без контекста сотрудник может прочитать как ошибку. Стоит проверить, что это именно текущее маркетинговое название.

## Missing Checks

- Не проверял `bot_template_registry.json` поштучно на все 173 факта — выборочно подтверждён `installment.provider` и набор `discounts.*`. Полная сверка `fact_id` ↔ template ↔ `bot_template_required=true` должна быть автоматическим тестом (если ещё нет).
- Не запускал `scripts/run_kb_semantic_review.py` на bot pack: путь не handoff release, gate в первую очередь рассчитан на `handoff_for_claude_and_team`. Бот-пак уже содержит готовый `semantic_review.json` от handoff-релиза. При желании можно запустить gate отдельно на handoff-папке — но это сделано прошлым semantic-ревью, итог `0 findings`.
- Не делал полнотекстовый просмотр всех 304 manager_only фактов на предмет случайно непустого `client_safe_text` — проверил только агрегатом (`allowed_for_client_answer=true` в manager_only = 0).

## Required Regression Tests Or Gates

1. Юнит-тест на builder: для всех фактов с `bot_template_required=true` запретить `client_safe_text` короче N символов или вида `«<бренд>: <ключ> — <значение>.»` без полноценной формулировки.
2. Юнит-тест на консистентность: множество `fact_id` в `bot_template_registry.templates[]` должно совпадать с множеством `fact_id` в `client_safe_facts_*.jsonl`, где `bot_template_required=true`. Различие — fail.
3. Тест на пустой `valid_until` в любом client-safe факте → fail (сейчас 0, нужно зафиксировать инвариант).
4. Тест на присутствие чужого бренда / 4 юр.лиц / номеров лицензий / КНД 1151158 в `client_safe_text` любого факта → fail.
5. Smoke-pack: либо реально 50 вопросов на бренд (≥40 на бренд против baseline), либо переименовать каталог.

## Recommended Next Step

1. v3.3 безопасно использовать как базу для **внутреннего пилота на сотрудниках** и для пилота на **лояльной подготовленной группе клиентов в режиме черновиков с обязательным одобрением менеджера** (это явно разрешено `bot_policy.yaml.bot_answer_self_rollout`).
2. Перед расширением до публичного трафика: закрыть Non-Blocking #1 (обрывочные тексты на стороне builder), Non-Blocking #3 (либо нарастить smoke50, либо переименовать), добавить тесты из раздела «Required Regression Tests Or Gates».
3. Зафиксировать в CLAUDE.md, что актуальный релиз — `v3_3` (сейчас в CLAUDE.md по-прежнему указан `v3_2`). Это решение Дмитрия, не Claude/Codex.
4. До декабря 2026 завести регламентный тикет на обновление фактов темы `contact`/`schedule` (короткий `valid_until=2026-12-31`).

## Summary
- formal_pass: да
- semantic_pass: да
- pilot_ready: да с условиями (внутренний + лояльные клиенты, режим черновика с одобрением менеджера; не публичный трафик)
- production_ready: нет
