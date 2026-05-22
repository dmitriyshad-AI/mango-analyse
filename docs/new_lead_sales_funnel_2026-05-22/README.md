# Продажная воронка нового лида для Telegram

Дата: 2026-05-22

Статус: `candidate` (кандидат на внедрение), не утвержденный клиентский поток.

## Назначение

Этот пакет описывает, как Telegram-боты Фотона и УНПК МФТИ должны вести нового лида от первого вопроса к понятному следующему шагу: подбору курса, записи, пробному формату, заявке или передаче менеджеру.

Пакет не меняет код, базу знаний, промпты, AMO, Tallanto, CRM, Telegram и `stable_runtime`.

## Источники

- `docs/PARALLEL_NEW_LEAD_SALES_FUNNEL_PROMPT_2026-05-22.md`
- `docs/PARALLEL_NEW_LEAD_SALES_FUNNEL_CONTEXT_2026-05-22.md`
- `docs/TELEGRAM_BOT_AUTONOMY_MATRIX_V1_2026-05-21.md`
- `docs/TZ_TELEGRAM_GOLD_ANSWERS_V3_INTEGRATION_2026-05-22.md`
- `docs/TZ_TELEGRAM_PILOT_FEEDBACK_HUMAN_CONTEXT_V2_2026-05-21.md`
- `docs/TELEGRAM_PILOT_EMPLOYEE_TESTING_GUIDE_2026-05-21.md`
- `product_data/knowledge_base/kb_release_20260520_v6_3_team_answers_employee_pack/START_HERE.md`
- `product_data/knowledge_base/kb_release_20260520_v6_3_team_answers_employee_pack/FOTON.md`
- `product_data/knowledge_base/kb_release_20260520_v6_3_team_answers_employee_pack/UNPK.md`
- `product_data/knowledge_base/kb_release_20260520_v6_3_team_answers_employee_pack/GOLD_ANSWERS.md`
- `product_data/knowledge_base/kb_release_20260520_v6_3_team_answers_employee_pack/MANAGER_ONLY.md`

## Файлы

- `NEW_LEAD_FUNNEL_MAP.md` - карта воронки нового лида.
- `BOT_INITIATIVE_RULES.md` - где бот держит инициативу и как не давить.
- `QUALIFYING_QUESTIONS_MATRIX.md` - безопасные уточняющие вопросы.
- `MANAGER_HANDOFF_RULES.md` - правила передачи менеджеру.
- `SALES_METRICS_SPEC.md` - метрики и события для измерения пользы.
- `SALES_ENABLEMENT_MATERIALS_PLAN.md` - материалы, которые стоит подготовить для продаж.
- `TEST_DIALOG_SCENARIOS.md` - 30 сценариев тестирования.
- `RISKS_AND_OPEN_QUESTIONS.md` - риски, спорные факты и вопросы Дмитрию/РОПу.
- `SEMANTIC_GATE_CANDIDATES.md` - кандидаты будущих смысловых gate-правил.
- `SEMANTIC_REVIEW.md` - смысловой аудит пакета.
- `CLAUDE_REVIEW_RESULT.md` - независимая внешняя проверка, если Claude CLI был доступен.

## Главные решения

1. Бот не должен быть просто справочником: после прямого ответа он делает один мягкий следующий шаг.
2. В одном клиентском ответе только один бренд: Фотон или УНПК МФТИ.
3. Максимум 1-2 уточняющих вопроса за сообщение; анкета из 10 вопросов запрещена.
4. P0/high-risk темы всегда передаются менеджеру: возврат, жалоба, конфликт, юридическая угроза, спорная оплата, чувствительные персональные данные.
5. Все новые формулировки, материалы и сценарии остаются кандидатами до утверждения Дмитрием/РОПом и превращения в тесты.

## Что можно передать в основной bot-диалог

- карту этапов воронки;
- матрицу уточняющих вопросов;
- правила инициативы;
- шаблон сводки менеджеру;
- метрики и события;
- тестовые сценарии как основу будущих автотестов.

## Что нельзя внедрять без проверки

- точные цены, даты, скидки, наличие мест и обещания;
- фразы про пробные занятия без отдельной проверки по бренду;
- сравнения Фотона и УНПК в клиентском ответе;
- автоответы по оплате, возвратам, договорам и жалобам;
- любые live-действия: запись, отправка ссылок на оплату, подтверждение оплаты, запись в CRM.
