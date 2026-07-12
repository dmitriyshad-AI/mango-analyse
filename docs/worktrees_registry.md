# Реестр worktree

Обновлено: 2026-07-12.

Источник факта: `git worktree list --porcelain` из общего gitdir Mango.

Правила:

- Один активный worktree закрепляется за одним исполнителем и одной крупной задачей.
- Исполнитель не переключает чужой worktree и не удаляет его `.codex_local`, runtime или незакоммиченные файлы.
- Удаление worktree, веток, тегов, runtime-данных и `git clean/gc/prune` требует отдельного точного предполёта и явного решения Дмитрия.
- Реестр подтверждает существование и защитную границу; ценность кода перед удалением проверяется отдельно.

## Активные worktree

| Worktree | HEAD | Ветка / состояние | Назначение | Решение |
|---|---:|---|---|---|
| `/Users/dmitrijfabarisov/Projects/Mango analyse` | `9e8fb3b` | `codex/tz135-direct-wow-tone` | Папка-хост общего `.git`, содержит параллельную работу. | Не переключать и не чистить. |
| `/private/tmp/mango_phase1_base_7da` | `7da115e` | detached | Временный base-worktree Phase1. | Удалять только отдельным решением. |
| `/Users/dmitrijfabarisov/Projects/Mango_britva_main` | `ca3626b` | `codex/faza1-test-speed-faza3-monolith-cleanup` | Трек чистки и ускорения тестов. | Не трогать. |
| `/Users/dmitrijfabarisov/Projects/Mango_britva_wave0_merge` | `3cf3f66` | `codex/britva-wave0-tz135-merge` | Параллельная интеграция wave0/tz135. | Не трогать. |
| `/Users/dmitrijfabarisov/Projects/Mango_calls_tzv_20260712` | `d653cce` | `codex/tzv-calls-schedule-brand` | Завершённая ветка расписания звонков и brand_evidence. | Не чистить до завершения интеграции. |
| `/Users/dmitrijfabarisov/Projects/Mango_email_pipeline_restore` | `b415ef7` | `codex/email-pipeline-restore` | Параллельный почтовый/Timeline трек. | Не трогать. |
| `/Users/dmitrijfabarisov/Projects/Mango_frame_gate_3_defects` | `e1fbbba` | `codex/frame-gate-3-defects` | Параллельный трек frame-gate. | Не трогать. |
| `/Users/dmitrijfabarisov/Projects/Mango_integrate_d3_d4_20260712` | `c57e882` | `codex/integrate-d3-d4-20260712` | Чистое дерево слияния почты Д3 и звонков Д4. | Текущий write-worktree интеграции. |
| `/Users/dmitrijfabarisov/Projects/Mango_live_cca8aeb4_consolidated` | `cca8aeb` | detached, live | Фактический live-код. | Не трогать без отдельного live-ТЗ. |
| `/Users/dmitrijfabarisov/Projects/Mango_live_next_step_phase1b` | `2cc82b1` | detached, rollback | Точка полного отката live. | Не трогать. |
| `/Users/dmitrijfabarisov/Projects/Mango_m1_memory_exam_build` | `796c0b8` | `codex/m1-memory-exam-build-20260711` | Сборка экзамена памяти для M1. | Не трогать. |
| `/Users/dmitrijfabarisov/Projects/Mango_overclaim_part_a` | `a083174` | `codex/overclaim-part-a-focus-m1` | Параллельный трек overclaim. | Не трогать. |
| `/Users/dmitrijfabarisov/Projects/Mango_tallanto_freshness_pilot_20260712` | `4858f8e` | `codex/tallanto-freshness-publish-amo-pilot-20260712` | Параллельный пилот свежести Tallanto. | Не трогать. |
| `/Users/dmitrijfabarisov/Projects/Mango_timeline_main_consolidation` | `8c3d8a6` | `codex/mail-three-stage-worktree-hygiene` | Завершённая почтовая цепочка Д3. | Не чистить до завершения интеграции. |
| `/Users/dmitrijfabarisov/Projects/Mango_wave0_refresh` | `636c52c` | `codex/wave0-refresh-20260710` | Параллельный трек wave0 refresh. | Не трогать. |
| `/Users/dmitrijfabarisov/Projects/Mango_z0_z1_latch_seats_20260712` | `60f8503` | `codex/z0-z1-latch-seats-20260712` | Параллельная задача latch/seats. | Не трогать. |

## Текущая задача

Проверенное слияние Д3 и Д4 выполняется только в
`/Users/dmitrijfabarisov/Projects/Mango_integrate_d3_d4_20260712`.
Live, rollback и параллельные worktree остаются неизменными.
