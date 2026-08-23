---
name: mango-development-process
description: "Обязательный процесс code-задач Mango: Graphify, доказательный inventory, полный Claude-контекст, вычисленные роли и закрытие Problem-ID."
---

# Процесс разработки Mango для Claude

Канонический процесс находится в
`.agents/skills/mango-development-process/SKILL.md`. Прочитай его целиком и
следуй машинным решениям `inventory_before_build.py` и `preflight.py`.

Твоя независимая роль: получить воспроизводимый context pack, проверить
существующего владельца реализации, путь вызова, соседние дубли и минимальный
корневой diff. Не восстанавливай контекст из чата и не разрешай код при
`decision=stop`, протухшем inventory или неполном receipt.

Разрешённый ТЗ функциональный smoke выполняй только тем же executor и в тех же
sandbox/escalated-границах, что и проверяемую реализацию. Read-only context
review не запускает тесты и не должен объявляться smoke-проверкой.
