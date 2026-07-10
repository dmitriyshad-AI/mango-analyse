> DONE 2026-07-07 03:36 | ветка codex/email-pipeline-restore | codex

> TAKE 2026-07-07 02:50 | ветка codex/email-pipeline-restore | codex

Ветка: codex/email-pipeline-restore
Зоны: src/mango_mvp/customer_timeline, scripts, tests, docs/DECISIONS_LOG.md, tasks/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest tests/ -q
Семантический-аудит: да

# ТЗ-D v2: реальные адаптеры ночной службы — по двум независимым D0-описям

Дата: 2026-07-07. Автор: Opus 4.8 (архитектор). Исполнитель: Codex 2. ЗАМЕНЯЕТ `2026-07-06_TZ_D_nightly_adapters_dlya_Codex2.md` (твоё ревью его BLOCKED — все 7 пунктов закрыты здесь). Очерёдность: ПОСЛЕ микро-захода (письма-контроль B1-B4); ревью этого ТЗ можно параллельно.

База фактов: две НЕЗАВИСИМЫЕ описи — `2026-07-06_D0_nightly_inventory_codex2.md` (твоя) и `2026-07-07_D0_nightly_inventory_OpusM4.md` (Opus M4; его находки: дневная серия Mango-API capture `mango_update_after_*` до 07-05 с УЖЕ работающим ASR-контуром; у mail/tallanto/wappi нет строк в `ingestion_cursors` вовсе). Числа сходятся по всем пересечениям. ⚠ ОДНО ПРОТИВОРЕЧИЕ ВЫВОДОВ: ты нашёл mail-handoff до 2026-06-30T16:56:24 (244 сообщения + 7 295 links) в `_external_handoffs/mail_archive_*`; Opus «свежее 06-19/06-21 не нашёл» (искал в других местах). Разрешается в Preflight (см. D0-шаг). Механизм required/optional уже есть (`nightly_service.py:31/:143-144/:165/:169`) — нужна МАТРИЦА и адаптеры.

## 0. Инварианты и СТОП
Запись ТОЛЬКО внутри `.codex_local/staging/**` (БД + seen-keys + манифесты снимков). Прод-БД — ни записи, НИ ЧТЕНИЯ в этом заходе (путь `customer_timeline_prod_20260621` не открывать). AMO/Tallanto/live-бот — 0 записей. Wappi: сетевой capture read-only разрешён (твой контур секретов), записи ТОЛЬКО в pending_attribution, НЕ в timeline_events до ручной привязки. Mango API: СЕТЬ В NIGHTLY ЗАПРЕЩЕНА — capture делает существующий дневной контур, nightly только ест готовое с диска. ASR НЕ запускать нигде. Массовые LLM-вызовы в nightly ЗАПРЕЩЕНЫ. Почта: IMAP/сеть в nightly запрещены — handoff-файлы производит внешний контур, nightly ест с диска. Литералы wappi: строго `wappi_telegram`/`wappi_max`. ПДн локально. «0 новых при успехе» = провал ТОЛЬКО если на диске лежат непроглоченные файлы (иначе честный «нет нового»). Числа отчёта = SQL.

## 0-бис. Preflight (первым, до кода)
Регистрация задачи (`task_move.py --take`), PROJECT_NOW до HEAD, `preflight.py` зелёный; `PRAGMA quick_check` staging (обычный ro, БЕЗ immutable); ФИЗИЧЕСКАЯ проверка mail-handoff: `ls -la _external_handoffs/mail_archive_*` — есть ли файлы с данными после 06-19 до 06-30 (разрешение противоречия двух D0; если файлов нет — СТОП, доложить: D2-план строится на них); dry-run парса конфига службы.

## 1. МАТРИЦА required/optional (решение архитектора; валидируй на ревью)
REQUIRED (фейл блокирует latest):
- `calls_and_amo_incremental` — уже ok (курсоры 07-01/07-03), оставить; добить из твоего D0: exhaustiveness при page-cap для AMO (доказать полноту страниц или честный partial-маркер шага). СЮДА ЖЕ (в REQUIRED, не в D5): влив ГОТОВЫХ обработанных call-сводок из дневной серии `mango_update_after_*` (07-02…07-05 и далее) — курсор звонков обязан двигаться от 07-01 (это критерий приёмки (г)).
- `mail_archive_incremental` — НОВЫЙ адаптер (сейчас adapter_todo): источник — daily archive handoffs (`_external_handoffs/mail_archive_*`; по твоему D0 на диске есть до 2026-06-30: 244 сообщения + 7 295 links — подтверждается Preflight'ом), стартовый курсор `2026-06-19T14:53:27+00:00` (= max(event_at) mail_archive_stage2), дедуп по `message_sha256` С УЧЁТОМ двух source_ref-контуров (a2v3_mail + mail_stage2 — известное задвоение клиента 20/01!), новые письма БЕЗ LLM-выжимки → статус «нужна выжимка позже». Плюс mail-freshness-алерт в манифест снимка («последний handoff старше N дней») — иначе после доедания до 06-30 required-почта молча жёлтеет навсегда, и «0 новых» неотличим от поломки внешнего контура.
OPTIONAL (честный skip с причиной в манифесте latest):
- `tallanto_money_incremental` — свежих выгрузок нет (заморожен 21.05; выгрузку делает человек): шаг optional + алерт «Tallanto freshness > 30 дней»; стартовые курсоры зафиксировать значениями: tallanto_snapshot `2026-05-21T08:59:36+00:00`, tallanto_crm_call `2026-06-04T16:54:54+00:00`; при появлении выгрузки — включение отдельным решением.
- `wappi_history_incremental` — pending-режим (см. §0): capture → pending_attribution; счётчики в отчёт; в timeline не пишет.
- `mango_api_freshness` — БЕЗ СЕТИ (см. §0): влив готовых сводок перенесён в REQUIRED calls-шаг; этот optional-шаг = только мониторинг свежести дневной серии `mango_update_after_*` (алерт «capture старше N дней») + persisted seen-keys для идемпотентности учёта.

## 2. Блоки
D1. Матрица в конфиг службы + стартовые `ingestion_cursors` для mail/tallanto/wappi (сейчас строк нет вовсе — из D0) + per-source_ref уникальность против пропусков (твой пункт 7 из ревью v1).
Тест: конфиг читается; required-фейл → latest_published=false (юнит уже есть — расширить на матрицу); курсоры созданы с задокументированными стартами.
D2. mail_archive_incremental по правилам §1. Тест: прогон на handoff-срезе → max(event_at) > 2026-06-19; ре-ран того же среза = 0 новых; дедуп-юнит на двойной source_ref одного message_sha256; 0 LLM-вызовов (счётчик).
D3. tallanto optional + алерт. Тест: без свежей выгрузки шаг skip с причиной, latest публикуется; юнит алерта.
D4. wappi pending-capture. Тест: raw → pending_attribution, timeline_events до/после = 0 дельты; литералы верные; ре-ран = 0 новых pending-дублей.
D5. mango_api_poll по §1. Тест: seen-keys идемпотентность; 0 новых ASR-транскриптов от службы.
D6. Полный прогон службы на staging: required зелёные → `latest_published=true`, манифест снимка (счётчики по source_system до/после, sha) + отчёт. Тест: повторный полный прогон сразу после — 0 новых событий везде.

## 3. Приёмка (мой регрейд R1 по сырью)
(а) счётчики каждого REQUIRED-источника выросли соответственно отчёту (мой SQL до/после; optional при честном skip не растут); (б) ре-ран каждого адаптера = 0; (в) почта: max(event_at) mail stage2 ≥ 2026-06-30 (по handoff-данным); (г) события локальных звонков продолжают приезжать (курсор двигается от 07-01); (д) от службы 0 новых ASR-транскриптов; (е) latest-манифест опубликован только при зелёных required; плюс pytest зелёный (не хуже 4084) и твой аудитор PASS. СТОП-условия из §0 — без исключений.

---
## ПРОМТ ДЛЯ CODEX 2
Возьми ТЗ `Foton/2026-07-07_TZ_D_v2_nightly_adapters_po_dvum_D0_dlya_Codex2.md` — это переработка ТЗ-D по твоему же BLOCKED-ревью + две независимые D0-описи (твоя и Opus M4 — сходятся; его находки про дневной Mango-capture с ASR и отсутствие курсоров mail/tallanto/wappi учтены). Дай ревью (BLOCKED/PASS + правки; особо: матрица §1 — согласен ли с required/optional и стартовыми курсорами; D5 — не задвоит ли существующий дневной контур). Исполнение — после «финал» и после завершения твоего текущего микро-захода (письма-контроль). Рамки: только staging; wappi — pending-режим; ASR и массовые LLM — запрещены; ре-ран любого адаптера = 0 новых; числа = SQL.
