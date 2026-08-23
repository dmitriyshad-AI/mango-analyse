> DONE 2026-08-23 15:13 | ветка codex/context-reuse-gate-20260823 | codex

> TAKE 2026-08-23 14:43 | ветка codex/context-reuse-gate-20260823 | codex

# ТЗ A2: hard gate preflight и правила Claude/Codex

Ветка: codex/context-reuse-gate-20260823
Зоны: scripts/skills/inventory_before_build.py, scripts/skills/tz_lint.py, scripts/preflight.py, tests/test_preflight.py, tests/test_skills_top5_tools.py, .agents/skills/mango-development-process/SKILL.md, .claude/skills/mango-development-process/SKILL.md, .claude/agents/architect-auditor.md, AGENTS.md, CLAUDE.md, tasks/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_preflight.py tests/test_skills_top5_tools.py
Семантический-аудит: нет
Feature-ID: process.context_reuse_gate.preflight_v1
Problem-ID: process.context_loss_and_duplicate_builds
Изменение: extend
Ключевые-символы: parse_tz_header,run_preflight,inventory_before_build,mango-development-process
Ключевые-слова: prebuild evidence,hard gate,existing implementation,Claude process

Дата: 2026-08-23.
Предусловие: ТЗ A1 принято и влито.

## Контекст

Прочитать master-ТЗ и audit pack A1. Не писать новый parser: расширить
`parse_tz_header()` в `scripts/preflight.py`; `tz_lint.py` уже использует его.

## Реализация

1. Добавить `preflight --inventory <json>`.
2. Для любой ТЗ, чьи зоны включают `src/` или `scripts/`, требовать
   `Feature-ID`, `Problem-ID`, `Изменение`, ключевые символы/слова. Это правило
   действует и на старый inbox при следующем взятии в работу: отсутствие
   `Feature-ID` не является способом притвориться legacy.
3. Документационные/data-only ТЗ без code-зон остаются совместимыми.
4. Валидировать schema/version, feature/problem id, HEAD, fingerprint,
   generator command hash, непустые queries, coverage всех стадий, evidence,
   decision и selected owner.
5. Evidence path/sha перепроверять, а не доверять полям JSON.
6. Любое изменение HEAD/status/relevant untracked делает inventory протухшим.
7. `new` разрешать только с `ABSENT_PROVEN`, свежей картой и без unresolved.
8. Расширить существующий Claude skill и architect-agent порядком:
   `Graphify -> raw source -> all worktrees/refs/tasks -> decision -> code`.
9. Добавить session-local правило: внешний аудит/субагент с тем же
   `prompt_hash + files_hash + HEAD` повторно не запускается без причины.
10. В `AGENTS.md` и `CLAUDE.md` оставить только короткий инвариант и ссылку на
    skill, не копировать всю процедуру.

## СТОП

- предлагается второй header parser;
- JSON принимается без перепроверки evidence;
- старое code-ТЗ может обойти gate отсутствием Feature-ID;
- требуется persistent cache/service;
- требуется live, data scan, ASR или внешний write;
- бюджет более 150 строк нетестового кода.

## Приёмка

1. Тест-команда зелёная.
2. Code-ТЗ без новых полей блокируется независимо от возраста.
3. Data/docs-only legacy-ТЗ не ломается.
4. Сфабрикованный свежий JSON без coverage/evidence блокируется.
5. Смена HEAD, diff или relevant untracked блокирует старый inventory.
6. `reuse/extend/port` без owner и `new` без ABSENT блокируются.
7. Claude skill не разрешает код до явного решения.
8. Новых сервисов, зависимостей и флагов нет.
