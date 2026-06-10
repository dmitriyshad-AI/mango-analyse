# TZ-13 D4: merge TZ-12 + follow-up качества профилей

Дата: 2026-06-11.

## Результат

Статус: выполнено локально на основной машине. M1 и Яндекс-шина не использовались.

Live write не выполнялся: AMO, Tallanto, Wappi, Telegram/send, ASR, R+A не запускались. `stable_runtime` использовался только read-only для SQL-оценки и пересборки профилей из существующего master DB.

## Коммиты

- `dc6ee1ec` — `Merge TZ12 client history profile into main`
- `02dd2ae6` — `TZ13 RP1 mark duplicate child slots`
- `64b0fd32` — `TZ13 RP2 harden channel imports`
- `07c6947d` — `TZ13 RP3 tag historical channel brands`
- `77732520` — `TZ13 RP4 confirm quiet default and rerun sizing`

## РП-0

Ветка `codex/tz12-d4-client-history-profile` смержена в `main` отдельным merge-коммитом.

Проверки после merge:

- полный pytest после merge: `2958 passed, 5 skipped, 1 warning`;
- `src/mango_mvp/channels/` не изменён: diff пустой;
- channel smoke: `227 passed`;
- рабочее дерево TZ-13 после merge было чистым.

## РП-1: дубль-дети

Сделано:

- добавлена нормализация имени ребёнка по безопасному имя-токену и словарю уменьшительных;
- неоднозначные уменьшительные, например `Саша`, не мержатся автоматически;
- одинаковые child slots внутри одного профиля сшиваются в общий `child_key`;
- добавляется явный marker field `child_slot_merge_candidate` с `merge_candidate`;
- разные профили не пересекаются;
- повторная сборка идемпотентна.

Метрики реальной пересборки `tz12_working_batch3`:

- профилей с 2+ child slots до: `4640`;
- профилей с 2+ child slots после: `4471`;
- `merge_candidate_groups`: `817`;
- `child_slots_marked_merge_candidate`: `2077`;
- `child_slot_fields_rekeyed`: `4509`;
- `merge_candidate_markers_written`: `817`.

## РП-2: импорт-риски

Сделано:

- в TG/WA import report добавлен счётчик `ambiguous_phone_matches`;
- битая физическая JSONL-строка Telegram теперь пропускается со счётчиком `bad_jsonl_rows`, импорт не падает;
- `telegram_import_report.json` добавлен в `.gitignore`;
- NEG `git check-ignore telegram_import_report.json`: `rc=0`.

Исторические import reports в `tz12_working_batch3` были созданы до РП-2 и не пересобирались, чтобы не делать реимпорт. Новые счётчики закреплены кодом и тестами.

## РП-3: бренды каналов

Сделано:

- Telegram default brand: `unpk`;
- WhatsApp default brand: `unpk`;
- WhatsApp payload/metadata получает `channel_shared=true`;
- WhatsApp relevance tag хранится как `channel_shared:true`, потому что существующая нормализация тегов не допускает символ `=`;
- добавлен идемпотентный локальный ретро-скрипт `scripts/retrofit_channel_brand_tags_in_timeline.py`.

Реальный ретро-прогон по `tz12_working_batch3/customer_timeline.sqlite`:

- первый `--apply`: изменено `93968` строк;
- Telegram: `6950` events + `6950` chunks переведены `unknown -> unpk`;
- WhatsApp: `40034` events + `40034` chunks переведены `unknown -> unpk`, `channel_shared=true`;
- повторный `--apply`: `changed.total = 0`.

## РП-4: quiet_minutes и зона перепрогона

Сделано:

- `DEFAULT_QUIET_MINUTES = 30`;
- `detect_quiet_dialogs`, `refresh_from_journal`, CLI `--quiet-minutes` используют единый дефолт;
- тест подтверждает дефолт 30 и CLI override.

Read-only SQL-оценка зоны перепрогона:

- активных AMO-клиентов: `1303`;
- strong Tallanto student клиентов: `7298`;
- объединённая зона клиентов: `7965`;
- звонков в зоне: `46187`;
- с запасом 20%: `55425`.

Сам перепрогон не запускался.

## Обезличенные примеры после слияния дублей

Имена, телефоны и реальные profile_id не выводятся.

| profile_ref | marker | source_slots_marked | active_child_slots_after | grade | subject |
|---|---:|---:|---:|---|---|
| `8e324c4dc3` | `merge_candidate` | 2 | 1 |  | математика; физика |
| `a747f0df52` | `merge_candidate` | 3 | 4 | 6 | математика |
| `133f2b558d` | `merge_candidate` | 3 | 4 | 3 класс | физика |
| `98efdbbe2d` | `merge_candidate` | 2 | 1 | 9 класс | физмат |
| `e91b0bddec` | `merge_candidate` | 2 | 4 |  | математика |

## Артефакты вне git

- `/Users/dmitrijfabarisov/Projects/Mango analyse/product_data/customer_profiles/tz12_working_batch3/customer_timeline.sqlite`
- `/Users/dmitrijfabarisov/Projects/Mango analyse/product_data/customer_profiles/tz12_working_batch3/customer_profiles.sqlite`
- `/Users/dmitrijfabarisov/Projects/Mango analyse/product_data/customer_profiles/tz12_working_batch3/telegram_import_report.json`
- `/Users/dmitrijfabarisov/Projects/Mango analyse/product_data/customer_profiles/tz12_working_batch3/whatsapp_import_report.json`

`product_data/customer_profiles/` игнорируется git.

## Финальные проверки

- РП-1 tests: `19 passed`;
- РП-2 tests: `19 passed`;
- РП-3 tests: `18 passed`;
- РП-4 tests: `4 passed`;
- channel smoke: `227 passed`;
- full pytest: `2968 passed, 5 skipped, 1 warning`;
- `git diff --name-only 3fabecae..HEAD -- src/mango_mvp/channels channels`: пусто;
- tracked PII check по `product_data/customer_profiles`, raw Telegram/WhatsApp reports/sqlite: пусто.

Остаточный риск: `profile_build_report.json` в raw-директории остался старым TZ12 JSON-отчётом, потому что CLI пересборки печатает report в stdout и не перезаписывает файл автоматически. Фактическая SQLite-пересборка выполнена; актуальные метрики зафиксированы выше.
