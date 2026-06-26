# D1 Wappi live gate dry-run report

Дата: 2026-06-26  
Repo: `/Users/dmitrijfabarisov/Projects/Mango_main_intent_ff`  
Commit: `ca9b27d`  
Режим: `--once --dry-run`  

## Границы

- AMO live-write: `0`
- Отправки клиенту: `0`
- Tallanto/CRM write: `0`
- `stable_runtime`: не трогался
- Память клиента в черновике: `TELEGRAM_BOT_SAFE_CRM_CONTEXT=0`
- Auto-resolver: `DRAFT_LOOP_AUTO_RESOLVER=1`
- Codex tier: `MANGO_CODEX_SERVICE_TIER=flex`

## Артефакты

- Gate report: `/Users/dmitrijfabarisov/.mango_local/wappi_live_gate_20260626_133156/GATE_REPORT.md`
- Passport: `/Users/dmitrijfabarisov/.mango_local/wappi_live_gate_20260626_133156/passport_before_dryrun.json`
- Journal: `/Users/dmitrijfabarisov/.mango_local/wappi_live_gate_20260626_133156/journal.jsonl`
- Heartbeat: `/Users/dmitrijfabarisov/.mango_local/wappi_live_gate_20260626_133156/heartbeat.json`
- Daily report: `/Users/dmitrijfabarisov/.mango_local/wappi_live_gate_20260626_133156/daily_report_24h.json`
- Quality table: `/Users/dmitrijfabarisov/.mango_local/wappi_live_gate_20260626_133156/quality_table.csv`

## Итог dry-run

- Рассмотрено строк: `83`
- Черновиков построено в dry-run: `2`
- AMO-заметок записано: `0`
- Отправок клиенту: `0`
- Ошибок: `0`
- `pair_missing`: `74`
- `not_before_skipped`: `7`
- `pending_notes`: `0`
- `quarantined_pairs`: `0`

## Auto-resolver

Причины по `daily_report_24h.json`:

- `matched`: `30`
- `closed_lead`: `17`
- `max_phone_missing`: `10`
- `multi_active_lead`: `10`
- `brand_mismatch`: `4`
- `amo_chat_event_sequence_unconfirmed`: `1`
- `multi_contact`: `1`
- `username_only`: `1`

Важное поведение: в dry-run auto-resolver только логирует `matched` кандидатов и не сохраняет auto-pairs. Это штатный предохранитель: создание auto-pair стоит под условием `not dry_run`. В live-watch matched-чаты сначала получат auto-pair с `not_before_ts=now`, старые входящие будут пропущены, а заметки смогут создаваться только на будущие входящие.

## Решение гейта

Формальный dry-run gate пройден: ошибок, AMO write и отправок клиенту нет.

Включение постоянного Wappi watch с `--live-write` всё ещё требует отдельного явного разрешения Дмитрия, потому что это запись заметок в AMO.
