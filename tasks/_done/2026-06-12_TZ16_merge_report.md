# TZ16 Merge Report

Дата: 2026-06-12

## Итог

ТЗ-16 влито в `main` отдельным merge-коммитом после переноса артефактов в каноническую папку. Отдельное рабочее дерево `Mango-analyse-tz16` удалено штатной командой `git worktree remove`.

## Перенос артефактов

Источник до уборки:

`/Users/dmitrijfabarisov/Projects/Mango-analyse-tz16/product_data/customer_profiles/tz16_profiles_v7_20260612/`

Каноническое место:

`/Users/dmitrijfabarisov/Projects/Mango analyse/product_data/customer_profiles/tz16_profiles_v7_20260612/`

Перед переносом выполнен checkpoint/проверка SQLite:

- `customer_timeline.sqlite`: `PRAGMA wal_checkpoint(TRUNCATE)` -> `0|0|0`, `PRAGMA quick_check` -> `ok`
- `customer_profiles.sqlite`: `PRAGMA quick_check` -> `ok`
- `customer_profiles_idempotence.sqlite`: `PRAGMA quick_check` -> `ok`
- `customer_profiles_micro.sqlite`: `PRAGMA quick_check` -> `ok`

Сверка source -> target до удаления worktree:

- файлов в источнике: 13
- файлов в каноне: 13
- байт в источнике: 2 907 994 388
- байт в каноне: 2 907 994 388
- missing/extra/changed: `0/0/0`

Файлы в канонической папке:

```text
2054    anonymized_examples.json
94646272        customer_profiles.sqlite
94781440        customer_profiles_idempotence.sqlite
86016   customer_profiles_micro.sqlite
2718412800      customer_timeline.sqlite
32768   customer_timeline.sqlite-shm
0       customer_timeline.sqlite-wal
6971    rerun_tail_report.json
6972    rerun_tail_stdout.json
1437    source_hash_after.json
1437    source_hash_before.json
3296    step4_blacklist_microprobe.json
12925   summary.json
```

Проверка SQLite уже в канонической папке:

- `customer_timeline.sqlite`: `ok`
- `customer_profiles.sqlite`: `ok`
- `customer_profiles_idempotence.sqlite`: `ok`
- `customer_profiles_micro.sqlite`: `ok`

Проверка git ignore:

```text
.gitignore:141:product_data/customer_profiles/  product_data/customer_profiles/tz16_profiles_v7_20260612/summary.json
```

Вывод: raw/product артефакты перенесены, читаются и не добавлены в git.

## Merge

Pre-merge commit:

`53af2691e880e8b2f8be267e478f524598ebb135`

Merge commit:

`b309a951d73f793c1d6fa3e2a6ddb21430baebe1`

Diff pre-merge..merge:

```text
scripts/build_tz16_profiles_v7.py
scripts/compute_tz16_rerun_tail.py
src/mango_mvp/quality/non_conversation.py
src/mango_mvp/services/analyze.py
tasks/_done/2026-06-12_TZ16_D4_profiles_v7_and_rerun_tail_final_report.md
tasks/_done/2026-06-12_TZ16_step1_replant_cleanup_report.md
tasks/_done/2026-06-12_TZ16_step2_profiles_v7_report.md
tasks/_done/2026-06-12_TZ16_step3_rerun_tail_report.md
tasks/_done/2026-06-12_TZ16_step4_autoresponder_microfix_report.md
tests/test_analyze.py
tests/test_non_conversation_quality.py
tests/test_tz16_profiles_v7_build.py
tests/test_tz16_rerun_tail.py
```

Запретные пути в diff:

```text
<empty>
```

Не затронуты:

- `stable_runtime/`
- `channels/`
- `integrations/draft_loop.py`
- `amo_wappi*`
- tracked `product_data/`

## Тесты

Полный pytest на `main` после merge:

```text
3085 passed, 2 skipped, 1 warning in 44.21s
```

Предупреждение: системное `urllib3`/LibreSSL, не связано с ТЗ-16.

## Уборка worktree

Команда:

```text
git worktree remove /Users/dmitrijfabarisov/Projects/Mango-analyse-tz16
```

Результат:

```text
remove_rc=0
/Users/dmitrijfabarisov/Projects/Mango analyse  b309a95 [main]
worktree_removed=yes
```

## Статус рабочей папки

После merge и уборки остаются только не относящиеся к ТЗ-16 untracked-файлы, которые были оставлены без изменений:

```text
?? D1_audit_backlog/existing_clients/HANDOFF_live_card_v1_to_chief_architect_2026-06-12.md
?? tasks/_inbox_codex/2026-06-12_TZ16_merge_and_M1_tail_bundle_PROMPT_for_D4.md
```

Своих незакоммиченных изменений после отчёта быть не должно: этот файл отчёта коммитится отдельным коммитом поверх merge.
