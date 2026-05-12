# Transcript Quality Steps 2-3 Policy Report

Дата: 2026-05-09.

## Что выполнено

Закрыты шаги 2 и 3 из плана `TRANSCRIPT_QUALITY_POST_AUDIT_15_STEP_PLAN_2026-05-09.md`.

## Обновление policy: GPT-only

После обсуждения принято production-решение: не ждать обязательного консенсуса Claude/GPT в будущих прогонах. Основной контур проекта использует GPT, поэтому hard-gate auto-apply переводится в GPT-only policy:

- GPT decision должен быть `safe_apply`;
- deterministic safeguard должен оставаться `safe_apply`;
- deterministic label должен быть `non_conversation_high_confidence`.
- `policy_queue` должен быть `gpt_auto_apply`.

Claude остается внешним аудитом и источником регрессионных фикстур, но не блокирует production auto-apply.

Все остальные GPT/policy очереди блокируются:

- `gpt_keep_current`;
- `gpt_manual_review`;
- `reanalyze_required`.

Технически усилен backfill-валидатор:

- `src/mango_mvp/quality/transcript_quality_backfill.py` теперь распознает hard-gate GPT-policy кандидатов;
- строки с `gpt_keep_current`, `gpt_manual_review`, manual/protected-live deterministic flags не могут пройти в `planned_updates`/`apply`;
- `claude_decision` сохраняется в metadata backfill, если колонка есть, но не является hard blocker.

## Новые apply/review артефакты

Папка:

`stable_runtime/non_conversation_hard_gate_audit_package_200_20260509/`

Созданы/обновлены:

- `audit_gpt_apply_plan.csv` — 186 строк, которые GPT-only policy разрешает как auto-apply candidates;
- `audit_gpt_blocked_apply_plan.csv` — 14 строк, которые нельзя применять автоматически;
- `audit_gpt_non_auto_analysis.csv` — разбор GPT-only non-auto кейсов;
- `audit_gpt_policy.json` — формальная GPT-only policy для auto-apply;
- `audit_gpt_summary.json` — summary с `policy_auto_apply_allowed=186`, `policy_blocked=14`.

Также сохранены consensus-артефакты как справочная сверка:

- `audit_consensus_apply_plan.csv` — 185 strict GPT+Claude auto-apply;
- `audit_consensus_blocked_apply_plan.csv` — 15 non-consensus строк.

## Шаг 3: анализ 15 non-auto кейсов

Распределение 15 non-consensus кейсов:

- `consensus_keep_current`: 7;
- `manual_review`: 5;
- `disagreement_review`: 3.

Проверка против GPT-only policy:

- 1 кейс из 15 теперь разрешается, потому что GPT дал `safe_apply`, а deterministic safeguard зеленый;
- 14 кейсов остаются заблокированными: 13 `gpt_keep_current`, 1 `gpt_manual_review`;
- из заблокированных кейсов 12 уже объясняются deterministic v4 как `manual_review` или `keep_current`, 2 остаются GPT/policy blockers без расширения regex.

Ранее спорные кейсы:

- `hgate200_0079` — GPT safe, Claude manual; deterministic safe. В GPT-only policy теперь auto-apply candidate.
- `hgate200_0081` — GPT keep, Claude manual; deterministic safe. Есть короткое живое соединение с нецелевым собеседником и намерение дозвониться маме.
- `hgate200_0119` — GPT manual, Claude manual; deterministic safe. Service callback с ASR-loop/неразборчивой клиентской репликой.

Решение: не расширять deterministic regex под два оставшихся GPT-blocked кейса прямо сейчас. Это снизило бы полезность на очевидных no-live звонках и могло бы излишне отправить чистые автоответчики в manual review.

## Вывод по качеству

Текущий безопасный режим для staged backfill:

1. deterministic v4 находит кандидатов;
2. GPT-only policy задает production auto-apply;
3. apply script физически не пропускает GPT non-safe и deterministic manual/protected-live строки;
4. Claude-аудит используется как внешний контроль качества, но не как обязательный runtime gate.

Это быстрее и проще для продукта, чем обязательный Claude/GPT consensus, при сохранении ключевой защиты: GPT-safe решение всё равно не применяется без deterministic high-confidence guardrail.

## Проверка

Запущены targeted tests:

```text
PYTHONPATH=src python3 -m pytest tests/test_transcript_quality_backfill.py tests/test_transcript_quality_hard_gate_backfill_dry_run.py tests/test_non_conversation_quality.py
```

Результат:

```text
26 passed, 1 warning
```

Также выполнена компиляционная проверка измененных Python-файлов через `compile(...)` без записи pyc в системный cache.

## Следующий шаг

Перейти к шагам 4-5:

1. Решить, нужен ли отдельный `non_conversation_v5_consensus_safeguards`, или достаточно `v4_live_safeguards + hard_gate_consensus_policy_v1`.
2. Если нужен v5, добавлять только точечные правила с тестами, не расширять грубо regex.
3. Запустить новый full dry-run по 64 832 звонкам.
4. Сравнить с v4 dry-run: `5404 would_update`, `59428 unchanged`, `0 parse_errors`.
