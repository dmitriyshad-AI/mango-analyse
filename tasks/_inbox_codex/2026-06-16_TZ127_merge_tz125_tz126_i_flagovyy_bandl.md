# ТЗ-127 (после аудитора) — влить tz125+tz126 в main + собрать флаговый бандл для M1

- **Дата:** 2026-06-16. Заказчик: Дмитрий. Постановщик: Claude #1 (+ фокус-аудитор). Исполнитель: **D1** (мержит в main, собирает бандлы). Раннер-станция: M1.
- **Цель:** (1) закрыть в каноне Группу 4 и честную метрику; (2) собрать ОДИН бандл `mango_clean_<sha>` для прогона флагов OFF→ON, который заодно даёт свежее сырьё для валидации честной метрики.
- **Обе ветки я регрейднул — PASS.** tz125 — оффлайн (transcribe/config), боевого пути нет. tz126 — observe-only (метрика в раннере, транскрипты байт-в-байт).

## Жёсткие предпосылки (находки аудитора — без них НЕ начинать)
1. **Канон НЕ в главной папке.** Главная папка `Projects/Mango analyse` сейчас на `codex/tz119` + 40 грязных. `main` живёт в worktree `/Users/dmitrijfabarisov/Projects/Mango_main_tz121_merge` (6b93b77). **ВСЕ операции с main делать в этом worktree, не в главной папке.** 40 грязных файлов tz119 НЕ трогать, НЕ коммитить — они к ТЗ-127 не относятся.
2. Перед мержем: `git fetch` и убедиться `main == origin/main` (один источник правды).

## Шаги
### A. Влить обе ветки в main (в worktree `Mango_main_tz121_merge`)
1. `git checkout main`, `git merge --no-ff codex/tz125-finalize-group4` (a166a2c) — оффлайн Группа 4.
2. `git merge --no-ff codex/tz126-overhandoff-metric` (029cec7) — честная метрика (observe-only).
3. Конфликтов не ждём (tz125 — transcribe/config; tz126 — раннер; разные файлы). Если конфликт — СТОП/показать.
4. Полный pytest на main после влива зелёный (ожидаемо ~3300+). Запушить main.

### B. Интеграционная флаговая ветка
5. От нового main: `git checkout -b codex/measure-flags-honest`.
6. `git merge --no-ff codex/measure-tz122-tz123-tz124` (d59fdf27 = main+флаги tz122/124/123, все default OFF).
7. **Проверено мной:** правки раннера tz126 (строки 3541+/3733+) и d59fdf27 (1780–3242) НЕ пересекаются → слияние раннера чистое. Подтвердить, что в итоговом раннере есть И `_classify_handoff_bucket`/`over_handoff.buckets` (tz126), И флаговые правки. Если конфликт в раннере — разрулить, сохранив ОБА.
8. Полный pytest зелёный (вкл. `test_telegram_dynamic_client_sim.py` с бакет-тестами + tz122/123/124 тесты).

### C. Собрать бандл
9. `python3 scripts/build_mango_clean_bundle.py --repo <worktree codex/measure-flags-honest> --kb-snapshot product_data/knowledge_base/kb_release_20260612_v6_7_staging_r4_1/kb_release_v3_snapshot.json`
   - Скрипт копирует ВЕСЬ `git ls-files` → раннер `scripts/run_telegram_dynamic_client_sim.py` входит автоматически (overlay не нужен).
   - **Снапшот пробросить ЯВНО** (`v6_7_staging_r4_1`) — он же был у прошлого прогона d59fdf27 (подтверждено по его BUNDLE_INFO); бандл снапшот НЕ пинит, M1 берёт из своего product_data → честность держится на явном флаге, не на дефолте.
   - Наборы попадут идентичными прошлому прогону (autonomy_personas_unpk_20260613 + gain_nabor_20260615 + real_questions_20260531) — это гарантирует сам мерж d59fdf27 (тот же blob), не случайное совпадение.
10. Положить бандл в `~/Yandex.Disk.localized/OpenClaw/Actual Mango Tests/`, последним записать `manifest.json` (скрипт это делает сам).

### D. Прогон на M1
11. Один и тот же бандл: прогон **OFF** (флаги выкл), затем **ON** (tz122+tz124+tz123 вкл). Те же наборы/снапшот в обоих.
12. Команда: v2 ON, snapshot v6_7_staging_r4_1, memory codex low, semantic codex medium, **--parallel 4**, allow_default_autonomy temp, pilot_gold_v1, судья v9.

## Что проверить ПЕРЕД прогоном (моя приёмка-0)
- Грепнуть в коде интеграционной ветки, что флаги tz122/124/123 реально `default OFF` (не верить названию ветки).
- Подтвердить, что хунки tz126 сидят только в метрик-функциях (`_over_handoff_metrics`/`build_summary`/`_classify_handoff_bucket`), а не в `run_one_dialog`/генерации ответа.

## Моя приёмка (после прогона, по сырью)
1. **NEG-паритет:** `diff` `dynamic_dialog_transcripts.jsonl` OFF-прогона против прошлого C0 на том же снапшоте/наборе — поведение бота не изменилось (observe-only метрика + флаги OFF ничего не двигают). Сверяю транскрипты, НЕ сводку судьи.
2. **Эффект флагов OFF→ON:** over_handoff↓; P0/бренд-разделение/выдумки (hard-gate) держатся; per-dialog flip «ответ↔уход».
3. **Честная метрика на OFF-прогоне:** `over_handoff.buckets` — closing≈39%, legitimate(+disputed_p0)≈10%, upsell_miss≈15-20%, FN-дожима=0.

## Границы
Только Codex (без ключей). main НЕ ломать, дивергенцию не плодить (мержи в worktree main). Живой пилот не трогать. После сборки — `git worktree prune` для prunable-копий. Решение о включении флагов в пилот — отдельно, после моего регрейда (одно изменение за раз).
