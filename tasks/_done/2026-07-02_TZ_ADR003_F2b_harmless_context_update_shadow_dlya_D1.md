> DONE 2026-07-02 03:30 | ветка codex/adr003-semanticframe-migration | codex

> TAKE 2026-07-02 03:14 | ветка codex/adr003-semanticframe-migration | codex

# ТЗ ADR-003 Ф2b: найти реальный рычаг автономности после NO-GO Ф3 price

Ветка: codex/adr003-semanticframe-migration
Зоны: scripts/report_adr003_overhandoff_levers.py, tests/test_report_adr003_overhandoff_levers.py, audits/_inbox/, docs/worktrees_registry.md, tasks/_running/2026-07-02_TZ_ADR003_F2b_harmless_context_update_shadow_dlya_D1.md
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_report_adr003_overhandoff_levers.py tests/test_report_adr003_semantic_frame_eval.py tests/test_direct_path_semantic_frame_shadow.py
Семантический-аудит: да

Дата: 2026-07-02.

Исполнитель: D1/Codex.

Статус: план следующего безопасного шага, после свежего M1 `36ea110`.

## Основание

Свежий M1-прогон `36ea110` показал:

- Ф2 self-answer shadow безопасна: `route_text_diff=0`, `too_confident=0`, `p0/money/operational lowering=0`, `partial_freshness=0`.
- Но `would_demote_count=0`: текущий self-answer gate не нашёл ни одного хода для понижения.
- Ф3 price active = NO-GO: price почти весь уже self, а оставшиеся price/platform-price не проходят confidence/freshness.
- Реальный over-handoff в gold:
  - safe/self строк: `32`;
  - уже self: `21`;
  - handoff: `11`;
  - из них `manager_only=8`, `draft_for_manager=3`.
- Первичная гипотеза D1 была: harmless `context_update` / ack / status / closing.
- Регрейд Claude #1 по свежему сырью поправил гипотезу: основной реальный over-handoff — frame путает безопасную справку "существует ли курс/лагерь/группа/формат для класса X" с `check_availability/enroll`, то есть с живыми местами или записью.
- Поэтому главный blocked-класс: safe existence/format reference без подтверждённого exact fact. Его нельзя активировать без отдельного anti-fabrication / fact-verification слоя.
- Harmless ack/status остаётся отдельным малым подклассом с честной цифрой.

Сырьё анализа:

- M1 run:
  `/Users/dmitrijfabarisov/Yandex.Disk.localized/OpenClaw/Actual Mango Tests/adr003_f2_clean_36ea110_20260702/runs/adr003_f2_self_answer_shadow_36ea110`
- локальный разбор D1:
  `/Users/dmitrijfabarisov/Claude Projects/Foton/2026-07-02_ADR003_F2_real_lever_local_analysis_dlya_Dmitry_i_Claude.md`

## Цель Ф2b

НЕ менять поведение бота.

Построить новый shadow-измеритель/report-only слой, который честно показывает:

1. какие safe `manager_only` / `draft_for_manager` ходы являются справкой "существование/формат/пригодность" и должны оставаться blocked до отдельного anti-fabrication / fact-verification решения;
2. какие safe handoff ходы похожи на harmless context/update/ack/status и могли бы обсуждаться как route-only только после semantic-reggrade;
3. какой слой поставил handoff: `conversation_intent_plan`, `answer_contract`, message_type guard, low confidence, missing facts, safety flags.

Ф2b должна ответить: **есть ли реальный безопасный route-only рычаг автономности, отличный от price.**

Ф2b НЕ является разрешением на active. Особенно: `manager_only` здесь диагностируется только как источник over-handoff/root-cause. Любой будущий active по `manager_only` требует отдельного ADR/«да» Дмитрия и пересмотра железного правила исходного ТЗ.

## Жёсткие ограничения

- Ничего не включать в профиль.
- Live/Telegram/AMO/Tallanto/CRM не трогать.
- Route/text текущего бота не менять.
- Active-понижение не делать.
- P0-preblock/floor не трогать.
- `manager_only` в runtime не понижать.
- Не добавлять regex-понимание.
- Не трогать мораторий:
  - `tests/fixtures/adr003_*snapshot.json`
  - `CHANNEL_REGEX_BUDGET`
  - `docs/ADR003_REGEX_UNDERSTANDING_MORATORIUM.md`
- Всё за default-OFF shadow-флагом или только в отдельном отчётном скрипте.
- Если есть сомнение между "сам" и "менеджер" — классифицировать как blocked.

## Что сделать

### Шаг 1. Read-only scorer/report

Сделать отчётный слой без изменения поведения:

`scripts/report_adr003_overhandoff_levers.py`

Входы:

- `--transcripts` путь к `ON/dynamic_dialog_transcripts.jsonl`
- `--gold` путь к `adr003_frame_gold_labels_20260701.jsonl` или gold report JSON
- `--out-dir`

Выходы:

- `adr003_overhandoff_levers_report.json`
- `adr003_overhandoff_levers_report.md`

Отчёт должен считать:

- total turns;
- gold safe/self rows;
- safe rows already self;
- safe rows in `draft_for_manager`;
- safe rows in `manager_only`;
- breakdown by topic/class;
- breakdown by route;
- breakdown by `message_type`;
- breakdown by `safety_flags`;
- breakdown by `answer_contract.route_reason`;
- breakdown by `frame.risk_class`, `frame.answerability`, `frame.requested_action`, `frame.confidence`;
- list of candidates grouped into:
  - `existence_format_needs_fact_verification_blocked` — справка "существует ли курс/лагерь/группа/формат/пригодность", blocked до подтверждённого факта;
  - `danger_adjacent_blocked` — safe-label ход рядом с P0/fabrication/payment/refund/legal/complaint, не чистый кандидат;
  - `harmless_context_ack_status_candidate` — только factless ack/status без справки, CTA, мест, записи, цены, дат, расписания, обещаний менеджера;
  - `safe_reference_without_exact_facts` — всегда blocked, не кандидат на route-only active;
  - `low_confidence_or_missing_facts_blocked`;
  - `p0_or_money_or_operational_blocked`;
  - `unclear_review_required`.

Важно: `current_p0_signal` не использовать как единственный запрет, потому что текущий scorer считает `route == manager_only` как P0-сигнал. Вместо этого отдельно смотреть:

- runtime shadow `guards.actual_p0`;
- safety flags with P0 markers;
- direct/model P0 metadata;
- gold `expected.must_handoff`.

Отчёт должен писать:

- `source_rev`;
- sha256 входных файлов;
- redacted/truncated examples, без полных клиентских строк;
- source evidence: `dialog_id`, `turn`, route, message_type, flags, frame fields, contract fields, но без ПДн.

### Шаг 2. Shadow-кандидат для harmless context/update

Если делать runtime shadow, то только metadata, не route/text:

`TELEGRAM_SEMANTIC_FRAME_CONTEXT_ACK_SHADOW`

В этом ТЗ runtime shadow не обязателен. Достаточно read-only scorer.

⚠ Важно: исходное ТЗ запрещает понижать `manager_only` в active. Поэтому Ф2b может **измерять** harmless `manager_only`, но НЕ объявляет его готовым к active. Для `manager_only` статус должен быть отдельным: `would_need_manager_only_policy_decision`, а не `would_demote`.

Shadow может поставить `would_allow_self_context_ack=true` только для будущего route-only active из `draft_for_manager`.

Для `manager_only` shadow может поставить только `would_need_manager_only_policy_decision=true`.

Условия для обоих статусов одинаковые:

- текущий route `manager_only` или `draft_for_manager`;
- `result.message_type in {"context_update", "non_question", "wait_for_more"}`;
- нет прямого вопроса клиента;
- frame:
  - `risk_class == "safe"`;
  - `must_handoff is False`;
  - `answerability == "answer_self"`;
  - `requested_action == "answer_question"` или отдельный безопасный `acknowledge_status`, если он уже есть в schema;
  - confidence >= 0.90;
- нет P0/floor/preblock/model_p0;
- нет money/payment/refund/legal/complaint;
- нет live availability / booking / enroll / send_payment_link / check_availability;
- нет missing facts, если итоговый текст содержит факт;
- если итоговый текст содержит любой бизнес-факт, allowed только при `all_exact_facts_fresh_client_safe=true`;
- partial freshness всегда blocked;
- если exact facts есть, все они fresh + client_safe + active brand;
- итоговый текст не содержит:
  - обещание менеджера свяжется;
  - бронь/места/запись;
  - цену/дату/расписание без exact facts;
  - внутренние слова "бот/ИИ/Codex/Claude";
  - PII echo.

Если любое условие не выполнено — `blocked` с reason.

Acceptance будущего active:

- `draft_for_manager` harmless context/ack можно будет обсуждать как Ф3b route-only после semantic reggrade.
- `manager_only` harmless context/ack нельзя активировать без отдельного решения Дмитрия, потому что это нарушает прежний инвариант Ф3 "понижать только draft_for_manager".

### Шаг 3. Тесты

Добавить точечные тесты только для scorer/shadow:

- harmless status:
  - "Спасибо, подумаем" / "Все получили" / "Аккаунт активировали, всё работает" → shadow candidate allowed;
- negative context/status:
  - "Аккаунт активировали, но доступ не работает" → blocked;
  - "Ссылку получили, но не открывается" → blocked;
  - "Спасибо, подумаем" → allowed только если ответ чистый ack, без мест/записи/актуальности;
- context with missing business fact:
  - "Расскажите про обе" без exact facts → blocked;
- manager action:
  - "Можно записаться?" / "забронируйте" / "есть места?" → blocked;
- P0/money:
  - "оплатили, доступа нет" / "верните деньги" / "ошибка в договоре" → blocked;
- brand unknown/cross-brand → blocked;
- exact facts partial freshness → blocked.

### Шаг 4. Локальный прогон без M1

Запустить scorer на свежих артефактах `36ea110`:

`/Users/dmitrijfabarisov/Yandex.Disk.localized/OpenClaw/Actual Mango Tests/adr003_f2_clean_36ea110_20260702/runs/adr003_f2_self_answer_shadow_36ea110`

Отчёт должен дать:

- сколько реальных harmless context/status кандидатов;
- сколько blocked и почему;
- есть ли хотя бы 3-5 устойчивых кандидатов для будущего Ф3b active route-only;
- список строк для Claude #1 semantic review.

### Шаг 5. Audit pack

Создать audit pack:

`audits/_inbox/adr003_f2b_overhandoff_levers_<timestamp>/`

Внутри:

- `implementation_notes.md`
- `changed_files.txt`
- `test_output.txt`
- `semantic_review.md`
- `risk_review.md`
- `backward_compatibility.md`
- JSON/MD отчёты scorer.

## Приёмка Ф2b

Ф2b = PASS только если:

- route/text live-поведения не меняются;
- scorer/reports воспроизводимы;
- найденные candidates не включают P0/money/booking/live availability;
- все candidates имеют понятный reason и source evidence;
- `current_p0_signal` не используется круговым образом;
- тесты зелёные;
- full pytest или согласованный targeted+collect-only зелёный;
- Claude #1 делает semantic reggrade candidates по сырью.

## Что НЕ делать

- Не готовить Ф3 price.
- Не делать active gate.
- Не понижать `manager_only` в runtime.
- Не называть `manager_only` кандидаты готовыми к active.
- Не делать no-fact self-answer active.
- Не добавлять regex-классы "спасибо/получили/подумаем" как бизнес-логику.
- Не менять P0/floor/preblock.
- Не менять профиль/live.

## Ожидаемый результат

Понятный ответ:

1. есть ли в текущем продстеке реальный безопасный over-handoff, который можно снять;
2. какие именно классы дают рычаг;
3. что можно готовить как следующий shadow/active phase;
4. что остаётся blocked до отдельной анти-выдумочной логики.
