# Realistic Agent Roadmap

Дата: 2026-05-05

Цель документа: зафиксировать практичный порядок развития Mango Analyse в ИИ-сотрудника для продаж без преждевременного ухода в инфраструктуру ради инфраструктуры.

## Короткий вывод

Направление из `TZ_FULL_VISION_2026-05-01.md` верное, но порядок Claude слишком инфраструктурный для текущей стадии. PostgreSQL, VPS, S3, live capture, bot и полноценный supervisor нужны, но максимальный ROI сейчас дает не это.

Сейчас первыми идут:

1. Довести исторический ASR до устойчивого distributed-процесса на M4 Max + M1 Pro.
2. Научиться надежно импортировать результаты второго Mac обратно.
3. После полной ASR-истории пересобрать contact-layer и ROP-пакеты.
4. Довести AMO writeback до staged production.
5. Потом делать live Mango capture, уведомления, action engine и bot.

## Текущее состояние на 2026-05-05

### Уже сделано или почти сделано

- M4 Max умеет готовить большие ASR-only батчи по датам.
- UI доработан: кнопки `Обновить статус`, `Параллельный pipeline старт`, `Параллельный pipeline стоп` вынесены наверх вкладки `Конвейер`.
- Завершен сентябрьский тестовый ASR-батч на 3000 звонков.
- Собран оставшийся сентябрьский ASR-only батч: `sep2025_asr_only_remaining_all_20260504`.
- На 2026-05-05 в этом сентябрьском батче: всего 7599, done 3471, in_progress 9, pending 4119.
- Собран M1 Pro пакет `external_m1_jan_mar_2025_asr_only_20260504`: январь-март 2025, 6070 звонков, ASR-only, переносимая папка с локальным `project_source`.
- M1 Pro пакет переведен в режим: 1 Whisper worker + 1 GigaAM worker, у GigaAM до 6 CPU threads внутри процесса.
- M1 test300 был возвращен и полезные результаты перенесены в `stable_runtime/external_m1_jan2025_test300_20260503`.
- Bulk Resolve+Analyze по накопленным ASR DB почти завершен.
- На 2026-05-05 осталось 27 actionable Analyze-хвостов: 15 в Jun-Jul-Aug, 5 в Apr-May, 7 в Night3000. Это не массовая очередь, а проблемные retry/fail хвосты.
- В AMO runtime уже есть shadow-run/writeback контур, blockers, safe-mode и guardrails.
- `agent_runtime` начал появляться в коде, но это пока не production action engine.

### Еще не сделано

- Нет универсального импортера M1-result DB обратно в рабочие DB.
- Нет полноценного dry-run отчета перед M1 import.
- Нет одной универсальной команды для сборки external worker pack на произвольный месяц/лимит/исключения.
- Contact-layer и ROP-пакет еще не пересобраны на максимально полной истории после новых Jan-Sep ASR.
- AMO field mapping не зафиксирован отдельным production-документом.
- Staged AMO writeback еще не доведен до режима dry-run -> staged-50 -> staged-300 -> full refresh на свежем contact-layer.
- Action engine L1-L4 не готов как production-система действий.
- Daily/Morning ROP digest пока не оформлен как регулярный артефакт.
- Live Mango API/capture, Telegram уведомления, PostgreSQL и sales bot остаются будущими этапами.

## Актуальный приоритет работ

### P0. Distributed historical ASR

Статус: в работе.

Что уже есть:

- M4 Max обрабатывает сентябрь 2025.
- M1 Pro получил Jan-Mar 2025 ASR-only пакет.
- Есть переносимые M1 инструкции, runtime setup, UI и pack-results.

Что нужно доделать:

1. Дождаться результата M1 Jan-Mar.
2. Импортировать M1 результат в рабочую систему без ручных переносов.
3. После September M4 и Jan-Mar M1 проверить общий ASR-gap по 2025.
4. Добрать оставшиеся месяцы/хвосты отдельными пакетами.

### P1. M1 result importer

Статус: не готов, это следующий инженерный блок.

Требования:

- Вход: zip/result folder с M1, внутри DB, transcripts, selected_calls, manifest.
- Match key: `source_filename`.
- Переносить: `transcript_manager`, `transcript_client`, `transcript_text`, `transcript_variants_json`, `transcription_status`, attempts/last_error where safe, transcript files.
- Не перетирать более свежий done без явного флага.
- Поддержать dry-run.
- Писать import report: found, missing, already_done, conflicts, updated, skipped, failed.

Минимальный CLI:

```zsh
python scripts/import_external_asr_result.py \
  --external-db /path/to/external.db \
  --target-db /path/to/target.db \
  --transcripts-dir /path/to/transcripts \
  --dry-run
```

### P2. External worker pack automation

Статус: частично сделано вручную, нужно обобщить.

Нужна команда, которая собирает пакет для второго Mac без ручной сборки:

```zsh
python scripts/prepare_external_asr_pack.py \
  --months 2025-10,2025-11 \
  --limit 2000 \
  --out stable_runtime/external_m1_... \
  --profile m1-pro-32gb \
  --copy-audio
```

Команда должна:

- исключать уже ASR done во всех текущих рабочих DB;
- исключать уже выбранное в активных external-pack, если нужно;
- копировать audio, не symlink, для переносимости;
- класть `project_source`, setup/run/status/pack scripts;
- генерировать prompt для Codex на втором Mac;
- генерировать manifest и контрольные CSV.

### P3. RA tail cleanup

Статус: массовый RA завершен, хвост 27.

Что сделать:

- Отдельно выгрузить 27 failed Analyze rows.
- Разделить ошибки на `turn interrupted`, `codex rc=1`, реальные плохие тексты.
- Для технических ошибок сделать controlled retry с меньшим параллелизмом.
- Для плохих/неразговорных звонков поставить `analysis_status=skipped/manual` с отчетом.

### P4. Rebuild contact-layer after ASR completion

Статус: ждать завершения текущих ASR и M1 import.

После ASR:

- пересобрать master calls;
- пересобрать master contacts;
- пересобрать AMO-ready contacts;
- убедиться, что длинные истории не обрезаются;
- пересчитать приоритеты с учетом даты анализа и давности последнего контакта.

### P5. Fresh ROP package

Статус: после P4.

Листы:

- Reopen
- Follow-up
- Manual review
- Top priorities
- Инструкция для РОПа
- Инструкция для менеджера

Обязательно дотянуть:

- live Tallanto context;
- UTM/source;
- историю звонков после закрытия;
- повторы по контакту;
- несколько сделок на один контакт;
- исключение нормальных `действующий клиент` из reopen.

### P6. AMO field mapping and staged writeback

Статус: foundation есть, production-документ и staged rollout не закрыты.

Нужно зафиксировать:

- какие AI-поля контакта пишем;
- какие AI-поля сделки пишем;
- какие AMO fields должны быть textarea/long text;
- какие поля никогда не трогаем (`Id Tallanto`, `Филиал Tallanto` и подобные);
- что пишем только если `writeback_allowed = true`;
- что пишем в контакт, что в сделку;
- как менеджер видит контекст сделки без перехода в контакт.

Staged rollout:

1. dry-run на свежем contact-layer;
2. staged-50;
3. staged-300;
4. full refresh только после отчета ошибок.

### P7. Action engine L1-L4

Статус: позже AMO writeback, но до live agent.

Порядок:

1. Описать action schema.
2. Policy table L1-L4.
3. Action log.
4. L3 approval queue.
5. Только потом автоматические задачи/уведомления.

### P8. Daily ROP digest

Статус: после свежего ROP-пакета и AMO staged writeback.

Сначала не Telegram, а CLI/Markdown/XLSX:

- что обработано;
- сколько новых hot/warm/reopen;
- кому срочно звонить;
- где менеджер не сделал follow-up;
- что записано в AMO;
- какие ошибки/блокеры.

### P9. Live Mango API/capture

Статус: отложено.

Сначала изучить Mango API и сделать polling. Webhook и S3 позже.

### P10. PostgreSQL, insight-layer, sales bot

Статус: не сейчас.

PostgreSQL нужен перед live multi-process production. Insight-layer и sales bot имеют смысл только после полной истории, стабильного contact-layer и доказанного AMO writeback.

## Что делать следующим прямо сейчас

Пока ASR продолжает работать:

1. Сделать `scripts/import_external_asr_result.py` с dry-run.
2. Сделать отчетный формат для M1 import.
3. Сделать `scripts/prepare_external_asr_pack.py`, чтобы следующий M1 пакет собирался одной командой.
4. Отдельно добить RA хвост 27 failed Analyze.
5. После завершения M4 September и возврата M1 Jan-Mar: импорт, затем общий ASR-gap отчет.
6. После закрытия ASR-gap: пересборка contact-layer и свежий ROP-пакет.

## Решение по актуальности

План актуален. Изменение только в статусе: часть distributed ASR уже реально сделана, но ключевой недостающий элемент теперь не очередной UI/батч, а надежный импортер внешних ASR-результатов и автоматизация external-pack. Без этого второй Mac останется ручным ускорителем, а не устойчивой distributed-системой.
