> DONE 2026-08-23 14:42 | ветка codex/context-reuse-gate-20260823 | codex

> TAKE 2026-08-23 13:57 | ветка codex/context-reuse-gate-20260823 | codex

# ТЗ A1: точный inventory существующей реализации

Ветка: codex/context-reuse-gate-20260823
Зоны: scripts/skills/inventory_before_build.py, tests/test_skills_top5_tools.py, tasks/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_skills_top5_tools.py
Семантический-аудит: нет
Feature-ID: process.context_reuse_gate.inventory_v1
Problem-ID: process.context_loss_and_duplicate_builds
Изменение: extend
Ключевые-символы: inventory_before_build,parse_worktrees_porcelain,stale_banner,load_output_manifest
Ключевые-слова: existing implementation,feature reuse,partial worktree,donor ref

Дата: 2026-08-23.
Приоритет: первый. ТЗ A2, B и C не начинать до зелёной приёмки A1.

## Контекст

Сначала прочитать master-дизайн:
`tasks/_inbox_codex/2026-08-23_TZ_CONTEXT_REUSE_GATE_CLAUDE_GRAPHIFY.md`.

Не строить второй сканер. Расширить существующий
`scripts/skills/inventory_before_build.py` и переиспользовать:

- `parse_worktrees_porcelain()` из `scripts/preflight.py`;
- `load_output_manifest()` и `stale_banner()` из
  `src/mango_mvp/graphify_structural.py`;
- текущий output manifest Graphify, не сам `graph.json`.

## Реализация

1. Сначала тестами воспроизвести текущие дефекты:
   - шумный Graphify даёт ложный `FOUND`;
   - untracked-функция в другом worktree не видна;
   - `build_project_inventory()` делает лишний массовый проход.
2. Удалить вызов `build_project_inventory()` из feature-inventory.
3. Graphify запускать первым отдельными точными запросами. Ревизию брать из
   structural manifest существующим helper. Graphify-кандидат без raw-проверки
   не считать реализацией.
4. Read-only проверить все worktree из `git worktree list --porcelain`:
   status, targeted `rg` в `src/`, `scripts/`, `tests/`, `.claude/`; не читать
   data/runtime/audio/mail/calls.
5. Проверить tracked, staged, modified, untracked, невлитые refs, tasks,
   свежие audit packs и decisions.
6. Классифицировать кандидатов только значениями из master-раздела A2.
7. Писать `prebuild_inventory.json` и краткий `.md`; вместо общего `FOUND`
   выдавать `decision=reuse|extend|port|new|stop`.
8. Обязать JSON содержать generator version/command hash и coverage стадий:
   Graphify, worktrees, raw rg, git refs, tasks, безопасные audit metadata,
   decisions.
9. В audit packs читать только manifest и безопасный whitelist master-ТЗ.
10. Dirty-код в зоне задачи или с совпавшим бизнес-термином без классификации
    даёт `dirty_code_unclassified` и `decision=stop`.

## Обязательные проверки

1. Точный tracked-владелец найден.
2. Modified и untracked кандидаты найдены.
3. Untracked D3/Wappi-код в основном worktree получает
   `PARTIAL_WORKTREE`, не шумный `FOUND`.
4. Donor branch получает `DONOR_REF`.
5. Исторически удалённый код не предлагается к восстановлению.
6. Общий keyword без raw-доказательства становится `FALSE_MATCH`.
7. Stale Graphify блокирует только `ABSENT_PROVEN`, но не raw-находку.
8. Сфабрикованный JSON без coverage/evidence не считается полным результатом.
9. Monkeypatch `sqlite3.connect` и чувствительных readers доказывает, что
   inventory их не вызывает.
10. Старый тест `status == "FOUND"` намеренно переписан под новый контракт.

## СТОП

- нужен второй сканер или новая зависимость;
- требуется читать клиентские данные/runtime;
- найден пересекающийся dirty worktree без решения владельцев;
- `new` выбран при существующем кандидате;
- нет raw-подтверждения;
- добавлено более 150 строк нетестового кода без СТОП-формата Бритвы;
- требуется live/ASR/M1/внешний write.

## Зафиксированный СТОП Бритвы и решение

После ядра точного поиска независимый аудит потребовал fail-closed покрытие
невлитых refs, staged rename, dirty symlink, намеренного удаления, конфликтующих
владельцев и воспроизводимого fingerprint. Полный diff скрипта стал
`+300/-71` строк и превысил исходный бюджет 150 строк.

Рассмотрены две минимальные опции:

1. Оставить только ядро в лимите, но разрешать ложный `new` на перечисленных
   обходах. Отклонено: это ломает саму бизнес-цель задачи.
2. Разделить результат на две атомарные логические волны без второго сканера:
   A1a — точный поиск и классификация; A1b — fingerprint и fail-closed защита
   refs/rename/symlink/removal. Выбрано: обе волны остаются в одном helper и
   одном контракте, зависимости/сервисы/флаги не добавляются.

Это явное расширение после СТОП, а не скрытое превышение бюджета. В audit pack
нужно отдельно указать объём diff и причину сохранения защитного усиления.

## Приёмка

1. Тест-команда зелёная.
2. Три сквозных класса master-ТЗ воспроизведены.
3. Повтор на неизменной поверхности детерминирован.
4. На этом Mac со свежей картой время измерено; при результате более 15 секунд
   есть профиль и отдельное решение.
5. Независимый ломатель не может начать `new` при untracked/donor-кандидате.
6. Audit pack содержит точный HEAD, diff, результаты тестов и risk review;
   inventory лежит внутри него как generated evidence.
7. Новых сервисов, флагов и зависимостей нет.
