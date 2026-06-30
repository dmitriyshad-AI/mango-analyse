# Реестр worktree и несведённых веток

Обновлено: 2026-06-16, ТЗ-130.

Правило: ветки с уникальными коммитами не удаляются без отдельного решения владельца. Для удаления влитых веток используется только `git branch -d`; `-D` запрещён.

## Активные несведённые ветки

| Ветка | Worktree | Тема | Уникальных коммитов к `origin/main` | СТОП-дата | Решение |
|---|---|---|---:|---|---|
| `codex/tz-email-timeline-bridge` | `/Users/dmitrijfabarisov/Projects/Mango_tz_email_timeline_bridge` | D4 мост письма+звонки в тестовую customer_timeline | 0 | 2026-06-25 | Жива до регрейда D4 по ТЗ 2026-06-21 |
| `codex/etap2-step1-address-book` | `/Users/dmitrijfabarisov/Projects/mango-tz33-perf` | Etap2 address book / fresh relink артефакты | 0 | 2026-06-25 | Зарегистрировано для preflight D4; не трогать в этом ТЗ |
| `codex/d7-amo-safety` | `/Users/dmitrijfabarisov/Projects/Mango_d7_amo_safety` | D7 AMO safety | 0 | 2026-06-25 | Зарегистрировано для preflight D4; не трогать в этом ТЗ |
| `codex/etap1-crm-card-assembler` | `/Users/dmitrijfabarisov/Projects/Mango_etap1_crm_cards` | Etap1 CRM card assembler | 0 | 2026-06-25 | Зарегистрировано для preflight D4; не трогать в этом ТЗ |
| `codex/f8-clean-defer-scenarios` | `/Users/dmitrijfabarisov/Projects/Mango_f8_axes_catalog` | F8 clean defer scenarios | 0 | 2026-06-25 | Зарегистрировано для preflight D4; не трогать в этом ТЗ |
| `codex/foton-next-step` | `/Users/dmitrijfabarisov/Projects/Mango_foton_next_step` | Foton next step | 0 | 2026-06-25 | Зарегистрировано для preflight D4; не трогать в этом ТЗ |
| `codex/tz147-p0-deep-output-carry` | `/Users/dmitrijfabarisov/Projects/Mango_tz147_p0_deep` | TZ147 P0 deep output carry | 0 | 2026-06-25 | Зарегистрировано для preflight D4; не трогать в этом ТЗ |
| `codex/tz148-env-isolation` | `/Users/dmitrijfabarisov/Projects/Mango_tz148_env_isolation` | TZ148 env isolation | 0 | 2026-06-25 | Зарегистрировано для preflight D4; не трогать в этом ТЗ |
| `codex/tz-uskoreniya-3-punkta` | `/Users/dmitrijfabarisov/Projects/Mango_uskoreniya_3_punkta` | Ускорения 3 пункта | 0 | 2026-06-25 | Зарегистрировано для preflight D4; не трогать в этом ТЗ |
| `codex/etap3-botsafe-layer` | `/Users/dmitrijfabarisov/Projects/Mango_etap3_botsafe` | Этап 3 Фаза 0 bot-safe слой customer_timeline | 0 | 2026-06-28 | Жива до регрейда bot-safe слоя |
| `codex/tz1-telegram-aprel` | `/Users/dmitrijfabarisov/Projects/Mango_tz1_telegram_aprel` | ТЗ-1 ingest апрельского Telegram-экспорта в тестовую customer_timeline | 0 | 2026-06-28 | Зарегистрировано для preflight D4; не трогать другие worktree |
| `codex/tz139-customer-timeline` | `/Users/dmitrijfabarisov/Projects/Mango_tz139_customer_timeline` | TZ139 customer_timeline Step 0 + Work A | 0 | 2026-06-25 | Жива до ревью Work A; потом решать Work B |
| `codex/tz135-direct-wow-tone` | `/Users/dmitrijfabarisov/Projects/Mango analyse` | TZ135 direct wow tone, основная папка сейчас не main | 4 | 2026-06-25 | Зарегистрировано для preflight; судьбу ветки решить отдельно |
| `codex/wappi-controlled-watch-observe` | `/Users/dmitrijfabarisov/Projects/Mango_wappi_controlled_watch_observe` | Wappi observe / controlled watch, отдельная наблюдательная петля | н/д | 2026-06-30 | Живая наблюдательная ветка; не трогать в ТЗ-153 |
| `codex/tz133-handoff-closing-metric` | `/Users/dmitrijfabarisov/Projects/Mango_tz133_handoff_closing` | TZ133 handoff closing metric | 1 | 2026-06-25 | Зарегистрировано для preflight; решить отдельно |
| `codex/tz137-adr002-direct-slots-fallback` | `/Users/dmitrijfabarisov/Projects/Mango_tz137_behavior_measure` | TZ137 ADR002 direct slots fallback | 1 | 2026-06-25 | Зарегистрировано для preflight; решить отдельно |
| `codex/tz138-analyze-sweep` | `/Users/dmitrijfabarisov/Projects/Mango_tz138_analyze_sweep` | TZ138 analyze sweep | 1 | 2026-06-25 | Зарегистрировано для preflight; решить отдельно |
| `codex/tz142-flagb-closing-fix` | `/Users/dmitrijfabarisov/Projects/Mango_tz142_flagb_closing` | TZ142 flag B closing fix | 2 | 2026-06-25 | Зарегистрировано для preflight; решить отдельно |
| `codex/tz34-child-escalation` | `/Users/dmitrijfabarisov/Projects/Mango_tz31_child_identity` | TZ34 child escalation | 3 | 2026-06-25 | Зарегистрировано для preflight; решить отдельно |
| `codex/foton-next-step-resolver` | `/Users/dmitrijfabarisov/Projects/Mango_foton_next_step` | Foton D8 deterministic next_step resolver | 0 | 2026-06-27 | Жива до отчёта и ревью |
| `codex/etap2-step1-address-book` | `/Users/dmitrijfabarisov/Projects/mango-tz33-perf` | Etap2 address book/perf worktree | н/д | 2026-06-27 | Зарегистрировано для preflight; судьбу ветки решить отдельно |
| `codex/d7-amo-writeback-safety` | `/Users/dmitrijfabarisov/Projects/Mango_d7_amo_safety` | D7 AMO writeback safety | н/д | 2026-06-27 | Зарегистрировано для preflight; судьбу ветки решить отдельно |
| `codex/etap1-crm-card-assembler` | `/Users/dmitrijfabarisov/Projects/Mango_etap1_crm_cards` | Etap1 CRM card assembler | н/д | 2026-06-27 | Зарегистрировано для preflight; судьбу ветки решить отдельно |
| `codex/f8-clean-defer-scenarios` | `/Users/dmitrijfabarisov/Projects/Mango_f8_axes_catalog` | F8 axes/catalog scenarios | н/д | 2026-06-27 | Зарегистрировано для preflight; судьбу ветки решить отдельно |
| `codex/tz147-p0-deep-output-carry` | `/Users/dmitrijfabarisov/Projects/Mango_tz147_p0_deep` | TZ147 P0 deep output carry | н/д | 2026-06-27 | Зарегистрировано для preflight; судьбу ветки решить отдельно |
| `codex/tz148-env-isolation` | `/Users/dmitrijfabarisov/Projects/Mango_tz148_env_isolation` | TZ148 env isolation | н/д | 2026-06-27 | Зарегистрировано для preflight; судьбу ветки решить отдельно |
| `codex/tz-email-timeline-bridge` | `/Users/dmitrijfabarisov/Projects/Mango_tz_email_timeline_bridge` | Email timeline bridge | н/д | 2026-06-27 | Зарегистрировано для preflight; судьбу ветки решить отдельно |
| `codex/tzA-amo-card-dry-run` | `/Users/dmitrijfabarisov/Projects/Mango_tzA_amo_card_writeback` | AMO card dry-run worktree | н/д | 2026-06-27 | Зарегистрировано для preflight; не трогать в этом ТЗ |
| `codex/tz-uskoreniya-3-punkta` | `/Users/dmitrijfabarisov/Projects/Mango_uskoreniya_3_punkta` | ТЗ ускорения 3 пункта | н/д | 2026-06-27 | Зарегистрировано для preflight; судьбу ветки решить отдельно |
| `codex/measure-flags-honest` | `/Users/dmitrijfabarisov/Projects/Mango_measure_flags_honest` | Флаговый бандл/замер TZ122+TZ123+TZ124 с честной метрикой | 12 | 2026-06-23 | Жива до флагового регрейда; потом влить/бросить |
| `codex/measure-tz122-tz123-tz124` | `/Users/dmitrijfabarisov/Projects/Mango_measure_tz122_tz123_tz124` | Предыдущая замерная ветка TZ122+TZ123+TZ124 | 11 | 2026-06-23 | Сравнить с `measure-flags-honest`, затем бросить/архивировать |
| `codex/tz118-group4-primary-d` | `/Users/dmitrijfabarisov/Projects/Mango_tz118_primary` | Группа 4 primary-D | 10 | 2026-06-23 | Решить: влить или бросить |
| `codex/tz116-offline-understanding` | `/Users/dmitrijfabarisov/Projects/Mango_tz116_offline` | Offline-understanding / трассы | 7 | 2026-06-23 | Решить: влить или бросить |
| `codex/tz123-tz124-remeasure` | `/Users/dmitrijfabarisov/Projects/Mango_tz123_tz124_remeasure` | Перезамер TZ123+TZ124 | 5 | 2026-06-23 | Решить после регрейда |
| `codex/tz106-real006-model-p0-on` | `/Users/dmitrijfabarisov/Projects/Mango_tz103_action_judge` | Замер real_006/model P0 | 3 | 2026-06-23 | Решить: оставить как историю замера или бросить |
| `codex/tz123-question-instead-of-handoff` | `/Users/dmitrijfabarisov/Projects/Mango_tz113_114_115_profile` | Вопрос вместо ухода | 2 | 2026-06-23 | Решить после оценки surface |
| `codex/tz122-wrong-intent-fact` | `/Users/dmitrijfabarisov/Projects/Mango_tz122_wrong_intent_fact` | Калибровка wrong_intent_fact | 3 | 2026-06-23 | Решить после регрейда |
| `codex/tz124-slot-anchor` | `/Users/dmitrijfabarisov/Projects/Mango_tz124_slot_anchor` | Slot anchor / bare grade | 2 | 2026-06-23 | Решить после регрейда |
| `codex/tz20-blacklist57` | `/Users/dmitrijfabarisov/Projects/Mango_tz20_blacklist` | TZ20 blacklist57 локальный прогон | 2 | 2026-06-23 | Решить отдельно от live AMO |
| `codex/block-a-deal-gold-expanded` | `/Users/dmitrijfabarisov/Projects/Mango_blockA_gold` | Expanded Block A deal gold measurement | 1 | 2026-06-23 | Решить: влить тестовый набор или бросить |
| `codex/tz135-direct-wow-tone` | `/Users/dmitrijfabarisov/Projects/Mango analyse` | TZ135 direct wow tone | 3 | 2026-06-23 | Зарегистрировано preflight ТЗ-34; решить отдельно |
| `codex/tz133-handoff-closing-metric` | `/Users/dmitrijfabarisov/Projects/Mango_tz133_handoff_closing` | TZ133 handoff closing metric | 1 | 2026-06-23 | Зарегистрировано preflight ТЗ-34; решить отдельно |
| `codex/tz137-adr002-direct-slots-fallback` | `/Users/dmitrijfabarisov/Projects/Mango_tz137_behavior_measure` | TZ137 ADR002 direct slots fallback | 1 | 2026-06-23 | Зарегистрировано preflight ТЗ-34; решить отдельно |
| `codex/tz138-analyze-sweep` | `/Users/dmitrijfabarisov/Projects/Mango_tz138_analyze_sweep` | TZ138 analyze sweep | 1 | 2026-06-23 | Зарегистрировано preflight ТЗ-34; решить отдельно |
| `codex/tz142-flagb-closing-fix` | `/Users/dmitrijfabarisov/Projects/Mango_tz142_flagb_closing` | TZ142 flag-B closing fix | 2 | 2026-06-23 | Зарегистрировано preflight ТЗ-34; решить отдельно |
| `codex/tz119-assumed-scope-guard-main` | нет активного worktree | TZ119 assumed-scope guard, грязь сохранена коммитами `a51444c` и `364322f` | 2 | 2026-06-23 | Решить после регрейда TZ119 |
| `codex/tz20-autoresolver` | `/Users/dmitrijfabarisov/Projects/Mango_tz20_autoresolver` | Живая AMO/autoresolver ветка | 1 | 2026-06-23 | Не трогать; отдельное ТЗ под контролем Дмитрия |

## Исключения после ТЗ-130

| Ветка | Причина | Решение |
|---|---|---|
| `codex/tz120-child-identity-off-on` | `git cherry origin/main` пустой, но worktree содержит untracked `tasks/_inbox_codex/2026-06-15_TZ120_nabor_child_identity_OFF_ON.md`; удаление worktree потеряло бы файл. | Не удалено. Нужно решить судьбу карточки, затем повторить `git branch -d`. |
| `codex/tz118-d-primary-clean` | `git cherry origin/main codex/tz118-d-primary-clean` вернул только `- b00b3bd...`, worktree был чист и удалён, но `git branch -d` отказал из-за отсутствия ancestry-merge. `-D` запрещён. | Ветка оставлена без worktree. Для удаления нужно отдельное решение владельца или ancestry-merge/тег-архив. |

## Detached worktree

| Worktree | HEAD | Проверка | Решение |
|---|---|---|---|
| `/tmp/mango_audit_wt` | `6b93b770` | commit содержится в `origin/main`, metadata была locked `initializing` | Разблокирован/prune, запись удалена |
| `/Users/dmitrijfabarisov/Projects/Mango_tz122_main_compare` | `dd00d65e` | commit содержится в `origin/main`, worktree clean | Удалён штатным `git worktree remove` |
