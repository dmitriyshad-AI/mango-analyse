# Transcript Quality Live Safeguards Audit

Дата: 2026-05-09

## Контекст

Claude провел полный аудит 200 hard-gate кандидатов и подтвердил общий вывод: voicemail/IVR/virtual-secretary hard-gate полезен, но перед массовым apply нужен protect-слой против редких живых диалогов.

Дополнительно выполнен независимый GPT/Codex-аудит того же пакета:

- `stable_runtime/non_conversation_hard_gate_audit_package_200_20260509/GPT_AUDIT_RESULT.md`
- `stable_runtime/non_conversation_hard_gate_audit_package_200_20260509/gpt_audit_decisions.jsonl`
- `stable_runtime/non_conversation_hard_gate_audit_package_200_20260509/gpt_audit_summary.json`

## Что исправлено в коде

Файл:

- `src/mango_mvp/quality/non_conversation.py`

Добавлены safeguard-паттерны:

1. `safeguard_transfer_after_live_dialogue` — менеджер предлагает перевести/соединить, клиент соглашается, а хвост звонка падает в voicemail/IVR.
2. `safeguard_long_client_live_turn` — длинная клиентская реплика с признаками живого диалога.
3. `safeguard_edtech_live_turn` — клиентская часть содержит EdTech-сигналы, и это не virtual secretary / не системный IVR.
4. `safeguard_proxy_parent_live_turn` — длинная реплика родителя/прокси с учебным контекстом.
5. `safeguard_sales_live_turn` — sales-call с клиентской репликой и учебными/коммерческими признаками.
6. `safeguard_third_party_ivr_after_live` — сначала сторонний IVR, но затем есть живой сервисный разговор.
7. `safeguard_live_opt_out` — короткий живой отказ/ошибочная заявка/неактуальность.
8. `safeguard_ambiguous_service_attempt` — сервисный callback с ASR-мусором, где безопаснее не auto-apply.

Дополнительно исправлено:

- payment/bank context внутри живого разговора больше не считается сторонним bank IVR;
- virtual secretary с фразами про AI не попадает под EdTech-live safeguard;
- расширен negative non-contentful context для случаев `разговора по существу не было`, `содержательного диалога не произошло`, `нецелевой`.

Версия guardrails в analyze обновлена:

- `non_conversation_v4_live_safeguards`

## Проверка на пакете 200

Файл результата:

- `stable_runtime/non_conversation_hard_gate_audit_package_200_20260509/gpt_after_safeguards_200.csv`

Распределение после правок:

- `safe_apply`: `188`
- `keep_current`: `5`
- `manual_review`: `7`

Это совпадает с Claude по главному числу auto-apply: `188` из `200`.

Отличие от Claude: я оставляю часть спорных live-кейсов в `manual_review`, а не в `keep_current`. Это намеренно консервативнее и безопаснее для массового backfill: такие звонки не будут автоматически перезаписаны в `non_conversation`.

## Полный dry-run v4 по корпусу

Папка:

- `stable_runtime/non_conversation_hard_gate_owner_dry_run_20260509_v4_live_safeguards/`

Режим:

- read-only;
- без LLM;
- без записи в рабочие SQLite DB;
- по `64 832` fully processed звонкам из coverage v5.

Результаты v4:

- rows scanned: `64 832`
- would_update: `5 404`
- unchanged: `59 428`
- parse errors: `0`
- protected live dialogues: `42 669`

Сравнение с v3:

- v3 would_update: `5 463`
- v4 would_update: `5 404`
- old v3 candidates blocked by v4 safeguards: `65`
- new safe candidates added by stricter negative/no-content detection: `6`
- net reduction: `59`

Файлы сравнения:

- `stable_runtime/non_conversation_hard_gate_owner_dry_run_20260509_v4_live_safeguards/v3_to_v4_blocked_by_safeguards.csv`
- `stable_runtime/non_conversation_hard_gate_owner_dry_run_20260509_v4_live_safeguards/v4_new_candidates_not_in_v3.csv`

## Тесты

Добавлены регрессионные тесты на:

- transfer-to-voicemail after live consent;
- third-party IVR tail after live service dialogue;
- short live opt-out / ошибочная заявка;
- ambiguous service callback with ASR junk.

Таргетный прогон:

```text
PYTHONPATH=src python3 -m pytest tests/test_non_conversation_quality.py tests/test_transcript_quality_hard_gate_backfill_dry_run.py
21 passed, 1 warning
```

## Вывод

Текущий v4 безопаснее v3: он сохраняет основную полезность hard-gate и блокирует главный найденный Claude/GPT риск — живые разговоры, загрязненные IVR/voicemail/ASR-хвостом.

Фактический auto-apply на исторические БД пока не выполнялся. Следующий шаг перед apply — сравнить с финальным Claude JSONL, если он будет положен именно в пакет 200, и затем собрать apply-plan только для v4 `would_update_candidates.csv`.
