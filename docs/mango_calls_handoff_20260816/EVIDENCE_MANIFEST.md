# Реестр доказательств пяти ТЗ

Дата составления: 15.08.2026.

Назначение: отделить измеренные факты от выводов и сделать повторную проверку возможной без телефонов, ФИО, текстов разговоров и аудио. Этот файл не является разрешением менять production.

## 1. Базовая версия кода

- Worktree: `<local-worktree>/.pipeline-interactive.20260814`.
- Branch: `codex/pipeline-interactive-20260814`.
- HEAD: `0ef7ff99d5b740c49adc9fa542107c55a3e14821`.
- `git status --porcelain`: пуст на момент проверки.
- Это evidence baseline. Перед реализацией нужно заново определить фактический service worktree по production config/launchd и доказать совпадение clean HEAD/config/env; правка другой копии запрещена.

Контрольные SHA-256 исходников baseline:

| Файл | SHA-256 |
|---|---|
| `src/mango_mvp/services/analyze.py` | `7f1efe5bbc8e5701f9fb0f755ab6ea7a5a439fb0fbb8366e2d4e68c118b048f9` |
| `src/mango_mvp/services/transcribe.py` | `f2f9899810292f5f5f380ee24a9c5d137dc76eccf425a03104fe4497594e62ed` |
| `src/mango_mvp/services/resolve.py` | `305906b4b08509e6cc1d059699f1ba3bb67b61f915962c7c27ea41bd9c38f304` |
| `src/mango_mvp/models.py` | `e18a1bd9a191897882334b11abac174ac88b27f5e86b2de1dcddced75aa39e3e` |
| `src/mango_mvp/db.py` | `ab43df9e49826d7fe6c3a522af4569c46130453e9715e96353a45602bb128e0c` |

Кодовые опоры, проверенные на этом SHA:

- `transcribe.py:1191` строит `dialogue_lines`; `transcribe.py:3461-3535` сохраняет хронологические строки и ролевой контракт.
- `resolve.py:315-440,1596-1615` уже умеет читать сохранённые `dialogue_lines`.
- `analyze.py:2511` берёт `call.transcript_text`, а не сохранённый диалог.
- `analyze.py:529-547` извлекает evidence только из ожидаемого формата строк.
- `analyze.py:590-596` форматирует время без явного общего `Europe/Moscow` для naive UTC.
- `analyze.py:1215-1274,1670-1704` не замыкает production guard неподтверждённых ролей.
- `models.py:16-73` содержит `CallRecord` и только общий `sync_status`, но не версию/хэш Google-проекции.

SHA-256 финальных плановых артефактов после cross-audit Codex:

| Артефакт | SHA-256 |
|---|---|
| `README.md` | `b1e4545d3360dac5c9f7783ae87b7844f9acd319095af9ca6364acd4e1551b74` |
| `TZ-01-canonical-dialogue-input.md` | `a4588606f65cc6a50b09492051fa3500c4219fc4f321f0bc61f107c6358b1744` |
| `TZ-02-role-attribution-guard.md` | `ad082c53bece61d5d03a654f58c0ef3721664cddb219b38d193eb05c6ed22de4` |
| `TZ-03-claim-evidence.md` | `81501d3bd41be63ed94c3ef5f0b86b26b86ad3c1701b1674f0d43bb5ea68ccde` |
| `TZ-04-safe-normalization-msk.md` | `1ef0f214f83cc0f7bb115aa923bb4170100d612b763bd71b190b62592a69e59a` |
| `TZ-05-versioned-google-publisher.md` | `b3abd09b6cbdb93de33fc352c9825ec64954b38744140ec5bdd2721d2d8e3731` |
| `M4_CALL_DATABASE_HANDOFF_PROMPT.md` | `205394f8520f7004fe97c7e1111adc04e3c51f543f73ac9d2b1a3644ddb9063f` |

`EVIDENCE_MANIFEST.md` не включает собственный SHA, чтобы не создавать рекурсивный контрольный хэш.

## 2. Снимок живой SQLite

- Источник: `<runtime>/working/mango_calls_pipeline.sqlite`.
- Время начала чтения: `2026-08-14T23:48:18Z` / `2026-08-15T02:48:18+03:00`.
- Доступ: `sqlite3 -readonly`, одна транзакция `BEGIN … COMMIT`, только агрегаты.
- Результат: `PRAGMA quick_check=ok`, `journal_mode=wal`.

Результаты одного согласованного DB-снимка:

| Метрика | Значение |
|---|---:|
| Всего звонков | 3 851 |
| Whisper `transcription_status=done` | 1 852 |
| Resolve terminal (`done/skipped/manual`) | 1 847 |
| Analyse done | 1 843 |
| Google `sync_status=done` | 1 817 |
| Analyse с непустыми `dialogue_lines` | 1 833 / 1 843 |
| `transcript_text` начинается с `CHANNEL_LEFT:` | 1 275 |
| `transcript_text` начинается с `MANAGER:` | 545 |
| Вход в ожидаемом timed-dialogue формате | 0 |
| `evidence=[]` | 1 843 / 1 843 |
| Непустой evidence | 0 / 1 843 |
| Роли trusted по строгому текущему признаку | 545 |
| Роли untrusted | 1 298 / 1 843 (70,4%) |
| Untrusted с `structured_fields.next_step.action` | 628 |
| Untrusted без `needs_review` | 982 |
| `history_summary` с точным UTC-префиксом SQLite | 1 843 / 1 843 |
| С соответствующим МСК-префиксом `UTC+3` | 0 / 1 843 |

Нормализованные SQL-запросы, которыми получены числа:

```sql
BEGIN;

SELECT count(*) AS total,
       sum(transcription_status='done') AS transcription_done,
       sum(resolve_status IN ('done','skipped','manual')) AS resolve_terminal,
       sum(analysis_status='done') AS analysis_done,
       sum(sync_status='done') AS sync_done
FROM call_records;

SELECT count(*) AS analysis_done,
       sum(json_type(transcript_variants_json,'$.dialogue_lines')='array'
           AND json_array_length(json_extract(transcript_variants_json,'$.dialogue_lines'))>0) AS dialogue_nonempty,
       sum(transcript_text LIKE 'CHANNEL_LEFT:%') AS prefix_channel,
       sum(transcript_text LIKE 'MANAGER:%') AS prefix_manager,
       sum(transcript_text GLOB '[[]??:??*[]] *:*') AS timed_prefix,
       sum(json_type(analysis_json,'$.evidence')='array'
           AND json_array_length(json_extract(analysis_json,'$.evidence'))=0) AS evidence_empty,
       sum(json_type(analysis_json,'$.evidence')='array'
           AND json_array_length(json_extract(analysis_json,'$.evidence'))>0) AS evidence_nonempty,
       sum(json_extract(transcript_variants_json,'$.role_mapping.confirmed')=1
           AND json_extract(transcript_variants_json,'$.role_mapping.manager_quality_allowed')=1) AS role_trusted,
       sum(NOT (coalesce(json_extract(transcript_variants_json,'$.role_mapping.confirmed'),0)=1
                AND coalesce(json_extract(transcript_variants_json,'$.role_mapping.manager_quality_allowed'),0)=1)) AS role_untrusted,
       sum(NOT (coalesce(json_extract(transcript_variants_json,'$.role_mapping.confirmed'),0)=1
                AND coalesce(json_extract(transcript_variants_json,'$.role_mapping.manager_quality_allowed'),0)=1)
           AND trim(coalesce(json_extract(analysis_json,'$.structured_fields.next_step.action'),''))<>'') AS untrusted_with_next_step,
       sum(NOT (coalesce(json_extract(transcript_variants_json,'$.role_mapping.confirmed'),0)=1
                AND coalesce(json_extract(transcript_variants_json,'$.role_mapping.manager_quality_allowed'),0)=1)
           AND coalesce(json_extract(analysis_json,'$.needs_review'),0)=0) AS untrusted_without_review
FROM call_records
WHERE analysis_status='done';

SELECT count(*) AS analysis_done,
       sum(substr(json_extract(analysis_json,'$.history_summary'),1,16)
           = strftime('%d.%m.%Y %H:%M',started_at)) AS exact_utc_prefix,
       sum(substr(json_extract(analysis_json,'$.history_summary'),1,16)
           = strftime('%d.%m.%Y %H:%M',datetime(started_at,'+3 hours'))) AS exact_msk_prefix
FROM call_records
WHERE analysis_status='done';

COMMIT;
PRAGMA quick_check;
PRAGMA journal_mode;
```

Ограничение: служба продолжает работу, поэтому более поздний запрос даст другие абсолютные количества. Сравнивать доли и выводы нужно либо внутри одной read-транзакции, либо с новым timestamp.

## 3. Снимок Google и проблема конспектов

Источник зафиксирован в основном аудите [MANGO_CALLS_PRODUCTION_AUDIT_AND_PLAN_2026-08-14.md](./MANGO_CALLS_PRODUCTION_AUDIT_AND_PLAN_2026-08-14.md), строки 33–50:

- timestamp: `2026-08-14 16:22:55 Europe/Moscow`;
- Google: 678 строк; SQLite `sync_done`: 678;
- 673 строки однозначно связаны по времени+телефону+длительности;
- 669 полных расшифровок имеют уникальный собственный hash;
- 620/673 конспектов совпали с текущим собственным Analyse;
- 53 — stale/иная историческая проекция собственного звонка;
- 0 доказанных точных конспектов другого звонка.

SHA-256 основного аудита: `18a9bf393fe4119668e3e1622758867e5edadb4d122cfdde2e787adb677c5817`.

Честное ограничение: immutable экспорт A:P и сырой per-row matcher к этому пакету не приложены. Поэтому числа 53/0 являются зафиксированным результатом того read-only аудита, а не независимо воспроизводимым сейчас snapshot bundle. ТЗ-05 именно поэтому требует stable ledger, полного live readback и отдельного identity gate до любой записи/сортировки.

## 4. Claude CLI и независимые проверки

- CLI: `Claude Code 2.1.223`.
- Использовалась именно CLI-команда вида `claude -p --model sonnet --effort high --tools '' --no-session-persistence`, а не браузер Claude.
- Каждый из пяти файлов ТЗ прошёл пять последовательных Claude-раундов; решения `принято/отклонено` записаны в журнале соответствующего файла.
- Затем отдельные агенты Codex выполнили архитектурный, ломающий, бизнес- и междокументный аудит. Их post-fix изменения честно не называются шестым Claude-раундом.

Ограничение воспроизводимости: полные raw prompt/output каждого Claude-раунда и их SHA не были сохранены отдельным owner-only bundle. Поэтому журналы в ТЗ — decision log, а не криптографический transcript аудита. Финальный текст после междокументных правок Claude повторно не видел; статус формулируется как «пять Claude-раундов + отдельный cross-audit Codex», без выдуманного финального Claude GO.

## 5. Внешние технические ограничения Google

- [`spreadsheets.batchUpdate`](https://developers.google.com/workspace/sheets/api/reference/rest/v4/spreadsheets/batchUpdate): подзапросы валидируются и применяются атомарно; из-за возможных действий соавторов после операции всё равно нужен readback.
- [Usage limits](https://developers.google.com/workspace/sheets/api/limits): 300 read/write requests в минуту на проект, 60 на пользователя/проект, рекомендуемый payload до 2 МБ, timeout обработки 180 секунд; при 429 нужен backoff.
- [Google Drive/Sheets file limits](https://support.google.com/drive/answer/37603): справка указывает 50 000 символов как границу содержимого одной ячейки при преобразовании в Sheets; проект использует 50 000 как собственный fail-closed предел и ничего не обрезает молча.

## 6. Что эти доказательства устанавливают, а что нет

Устанавливают:

- правильный хронологический диалог уже сохранён для подавляющего большинства Analyse; повторный ASR не нужен;
- текущий Analyse фактически получает другой, неудобный формат;
- текущий evidence не подтверждает ни один бизнес-факт;
- неподтверждённые роли массово проходят без обязательного review;
- UTC в конспекте — системный, а не единичный дефект;
- Google stale-проекция доказана; массовый cross-row shift на согласованном снимке не доказан.

Не устанавливают:

- что любая реплика ASR верна относительно аудио;
- что старый или новый ASR всегда лучше;
- что каждый stale/лексически слабый конспект фактически ложен;
- что доказательство из transcript равно доказательству из аудио;
- что пять ТЗ можно включать по отдельности в production.

Именно поэтому общий выпуск требует shadow, аудиовыборку, fail-closed role/evidence gates, exact Google readback и один совместимый cutover.
