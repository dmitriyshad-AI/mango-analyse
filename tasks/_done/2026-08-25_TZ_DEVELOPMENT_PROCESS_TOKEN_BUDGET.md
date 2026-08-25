> DONE 2026-08-25 12:31 | ветка codex/development-process-token-budget-20260825 | codex

> TAKE 2026-08-25 11:02 | ветка codex/development-process-token-budget-20260825 | codex

Ветка: codex/development-process-token-budget-20260825
Зоны: .agents/skills/mango-development-process/SKILL.md, scripts/make_audit_pack.py, tests/test_audit_pack_pii.py, docs/worktrees_registry.md, tasks/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_audit_pack_pii.py tests/test_skills_top5_tools.py
Семантический-аудит: нет
Feature-ID: feature.development_process_token_budget
Problem-ID: problem.development_process_unbounded_review_cycles
Исход: problem_closed
Closure-evidence: audits/_inbox/development_process_token_budget_final_20260825/implementation_notes.md
Изменение: extend
Ключевые-символы: run_claude_review,mango-development-process
Ключевые-слова: Claude review token budget repeated audits selected owner

# ТЗ: ограничить расход на повторные проверки

Добавить в существующий навык короткие правила: заранее назначать один
канонический вердикт и один проход каждой вычисленной роли, повторять только
после блокирующей ошибки или изменения проверяемого объекта, не
перезапускать полный набор без изменения кода, переиспользовать доказательства,
ограничивать число параллельных агентов и останавливать работу при падающей
ценности следующей проверки. Не ослаблять обязательные safety и semantic gates.
## Приёмка

- Правило короткое (не более 12 строк) и не повторяет общие инструкции.
- Указывает явные причины повторного аудита и тестов.
- Запрещает бесконечные циклы улучшений без новой ошибки.
- Для задачи без выбранного владельца prompt явно требует `SELECTED_OWNER: NONE`.
- Skill validator проходит командой:
  `/Users/dmitrijfabarisov/.codex/skill-venv/bin/python /Users/dmitrijfabarisov/.codex/skills/.system/skill-creator/scripts/quick_validate.py .agents/skills/mango-development-process`.

Inventory дал `decision=port` и выбрал `scripts/make_audit_pack.py` владельцем
исправления инструкции Claude. Его нужно править минимально. Правило бюджета
проверок добавляется в уже существующий канонический
`.agents/skills/mango-development-process/SKILL.md`: это второй фактический
владелец процесса, подтверждённый чтением сырого файла. Распознаватель Markdown
навыков не исправляется в этой задаче; для него оставить отдельный следующий
шаг `problem.inventory_markdown_skill_owner_detection`. Нестабильный отпечаток
из-за несвязанной грязной рабочей папки также оставить отдельным следующим
шагом `problem.inventory_cross_worktree_status_instability`. `.claude/skills/`
содержит тонкую обёртку со ссылкой на канонический `.agents` skill и остаётся
без изменений.

## СТОП

- Правка ослабляет обязательный safety, semantic review или preflight.
- Для результата требуется новый механизм, новый скрипт или зависимость.
