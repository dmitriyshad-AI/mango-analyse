# Архитектурный пакет: внутренний ИИ-сотрудник и будущий переход к продукту

Дата: 2026-05-22

Статус: архитектурное проектирование, не готовность к внешнему SaaS.

## Главный вывод

Сейчас правильная цель проекта - не строить SaaS, а стабилизировать внутреннего ИИ-сотрудника на базе двух Telegram-ботов:

- Фотон: `@foton_intellegence_bot`
- УНПК МФТИ: `@mipt_AI_bot`

Эти боты уже стали первым живым каналом. Но архитектурно Telegram должен остаться только одним из входов. Целевое ядро должно работать одинаково для Telegram, сайта, email, CRM-чата, Max/WhatsApp и будущего рабочего места менеджера.

Правильная траектория:

1. Стабилизировать текущий внутренний Telegram-пилот.
2. Отделить общее ядро ИИ-сотрудника от Telegram-обвязки.
3. Наладить постоянное хранение диалогов, решений, флагов и отзывов.
4. Зафиксировать разделение Фотон/УНПК как прототип будущей изоляции организаций.
5. Перенести запуск на отдельный MacBook M1 Pro или внутренний сервер.
6. Подготовить безопасный внешний demo-контур.
7. Думать о SaaS только после доказанной внутренней пользы.

## Текущий статус на 2026-05-22

Статус по `SAAS_READINESS_GATES`: `internal_pilot_ready` в процессе закрепления, до `internal_production_ready` ещё не дошли.

Что уже близко к `internal_pilot_ready`:

- два бренда разделены отдельными Telegram-ботами;
- база знаний v6.3 имеет `semantic_pass=true`;
- AMO/Tallanto используются только read-only;
- live-write в CRM запрещён;
- P0/high-risk темы должны уходить менеджеру;
- есть kill switch и runbook для текущего пилота.

Что ещё не закрыто:

- постоянное хранение диалогов и решений пока не является основным runtime-контуром;
- feedback сотрудников ещё нужно закрепить как ежедневный процесс;
- M1 Pro/service-контур ещё не основной способ запуска;
- внешний demo и SaaS пока преждевременны.

Если старые документы или `CLAUDE.md` указывают на более раннюю версию базы знаний, приоритет имеет актуальный release manifest и bot pack из текущего рабочего контура. Версию базы знаний нельзя брать из памяти диалога или старого документа без проверки текущего manifest.

## Важное про базу знаний

База знаний постоянно обновляется. На момент проектирования актуальный рабочий слой:

- `product_data/knowledge_base/kb_release_20260520_v6_3_team_answers/`
- bot pack: `product_data/knowledge_base/kb_release_20260520_v6_3_team_answers_bot_pack/`
- snapshot: `product_data/knowledge_base/kb_release_20260520_v6_3_team_answers/kb_release_v3_snapshot.json`
- gold-ответы: `product_data/knowledge_base/kb_release_20260520_v6_3_team_answers_bot_pack/GOLD_ANSWERS_FOR_BOT.md`
- правила gold-ответов: `product_data/knowledge_base/kb_release_20260520_v6_3_team_answers_bot_pack/gold_answer_rules.yaml`

Зафиксированные показатели v6.3:

- `facts_total`: 833
- `client_allowed_facts`: 472
- `semantic_pass`: true
- `blocking_findings`: 0

Архитектурное правило: ни один компонент не должен считать путь к базе знаний вечным. Всегда нужен `knowledge_release_id`, `snapshot_sha256`, `semantic_pass`, `active_brand` и возможность откатиться на предыдущую версию.

## Что создано в этом пакете

1. `AI_EMPLOYEE_ARCHITECTURE.md` - целевая архитектура ИИ-сотрудника.
2. `TELEGRAM_TO_CORE_SEPARATION.md` - как отделить Telegram-бота от общего ядра.
3. `DIALOGUE_DECISION_STORAGE_SPEC.md` - как хранить диалоги, решения, флаги и отзывы.
4. `TENANT_BRAND_ISOLATION_CONTRACT.md` - разделение брендов и будущая изоляция организаций.
5. `M1_SERVICE_RUNBOOK.md` - практичный runbook для MacBook M1 Pro.
6. `SAAS_READINESS_GATES.md` - критерии перехода от пилота к SaaS.
7. `ARCHITECTURE_RISK_REGISTER.md` - риски и анти-паттерны.
8. `IMPLEMENTATION_PHASES.md` - порядок реализации без преждевременного усложнения.
9. `QUESTIONS_FOR_DMITRY.md` - решения, которые нужны от владельца.
10. `P0_INCIDENT_PROTOCOL.md` - что делать при опасном ответе в пилоте.
11. `CLAUDE_ARCHITECTURE_REVIEW_REQUEST.md` - запрос на независимое ревью.
12. `CLAUDE_REVIEW_RESULT.md` - результат независимого ревью.
13. `CODEX_REVIEW_DECISIONS.md` - какие замечания приняты и как внедрены.
14. `semantic_review.md` - смысловой аудит архитектурного пакета.

## Источники

Прочитаны и учтены:

- `AGENTS.md`
- `docs/ROADMAP.md`
- `docs/DECISIONS_LOG.md`
- `docs/RUNBOOK.md`
- `docs/PROJECT_HISTORY_AND_PRODUCT_ROADMAP_2026-05-13.md`
- `docs/BUSINESS_MODULE_AUDIT_SAAS_PRODUCT_2026-05-19.md`
- `docs/BUSINESS_AUDIT_ACTION_REGISTER_2026-05-21.md`
- `docs/TELEGRAM_PUBLIC_PILOT_BOTS_RUNBOOK_2026-05-21.md`
- `docs/TZ_TELEGRAM_PILOT_FEEDBACK_HUMAN_CONTEXT_V2_2026-05-21.md`
- `docs/TZ_TELEGRAM_GOLD_ANSWERS_V3_INTEGRATION_2026-05-22.md`
- `docs/TELEGRAM_BOT_AUTONOMY_MATRIX_V1_2026-05-21.md`
- `docs/SEMANTIC_REVIEW_RULES.md`
- актуальные KB v6.3/gold-answer артефакты.

## Что не делалось

- Код не менялся.
- База знаний не менялась.
- Telegram runtime не менялся.
- Боты не перезапускались.
- ASR/R+A не запускались.
- `stable_runtime` не менялся.
- AMO/CRM/Tallanto не трогались.
- Клиентам ничего не отправлялось.

## Короткая рекомендация

Ближайшее развитие должно быть не “больше SaaS”, а “лучший внутренний ИИ-сотрудник”:

1. Постоянный журнал пилота.
2. Отчёт полезности ответов.
3. Ошибка бота -> тест или правка базы знаний.
4. Разделение брендов как обязательный P0-gate.
5. Перенос на M1 Pro только после 1-2 дней устойчивого пилота на текущем MacBook.
