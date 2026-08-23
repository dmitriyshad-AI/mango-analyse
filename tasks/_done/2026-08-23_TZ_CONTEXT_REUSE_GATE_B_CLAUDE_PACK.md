> DONE 2026-08-23 17:06 | ветка codex/context-reuse-gate-20260823 | codex

> TAKE 2026-08-23 15:15 | ветка codex/context-reuse-gate-20260823 | codex

# ТЗ B: воспроизводимый контекст Claude CLI

Ветка: codex/context-reuse-gate-20260823
Зоны: AGENTS.md, CLAUDE.md, scripts/make_audit_pack.py, scripts/preflight.py, tests/test_audit_pack_pii.py, tests/test_preflight.py, .agents/skills/mango-development-process/SKILL.md, .claude/skills/audit-pack-generator/SKILL.md, .claude/skills/audit-pack-generator/scripts/create_audit_pack.py, .claude/skills/mango-development-process/SKILL.md, docs/, tasks/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_audit_pack_pii.py tests/test_preflight.py
Семантический-аудит: нет
Feature-ID: process.context_reuse_gate.claude_pack_v1
Problem-ID: process.context_loss_and_duplicate_builds
Изменение: extend
Ключевые-символы: create_audit_pack
Ключевые-слова: Claude context pack,manifest sha256,read only review,audit PII masking

Дата: 2026-08-23.
Предусловие: ТЗ A1 и A2 приняты и влиты.

## Решение по бюджету Бритвы

Независимый review до кода подтвердил, что защищённые pack, verifier, unsigned
receipt и hard gate не помещаются в 150 строк одним безопасным патчем. Поэтому
это umbrella-ТЗ исполняется атомарными волнами B1–B5: безопасные path/privacy
примитивы; prompt+code-surface; builder; verifier+runner; preflight+wrapper.
Каждая волна укладывается в 150 строк нетестового кода; второго builder,
файла-модуля или сервиса не создаётся.

## Контекст

Прочитать master-дизайн и итоговый audit pack ТЗ A. Не создавать второй
pack-builder: расширить `scripts/make_audit_pack.py`.

## Реализация

1. Добавить явный режим Claude context review.
2. Обязательные файлы пакета:
   - `task.md`;
   - `prebuild_inventory.json`;
   - `git_context.txt`;
   - `manifest.json`, записанный последним.
3. Manifest содержит SHA-256 каждого предыдущего файла, HEAD, branch,
   worktree, feature/problem id и hash diff.
4. Пакет проверяется перед вызовом Claude; изменение байта даёт STOP.
5. Пакет не читает env, `~/.mango_secrets`, product_data, runtime, аудио,
   почту и клиентские выгрузки.
6. Полный Graphify-граф не копируется; selected evidence уже находится в
   inventory JSON.
7. Claude skill получает правило: функциональный smoke выполнять в той же
   sandbox/escalated-среде; `auth status` сам по себе не основание для login.
8. Одинаковый ручной повтор сверять по стабильному
   `HEAD + prompt_hash + files_hash`; timestamped `manifest_hash` хранить в
   receipt для целостности, но не включать в dedupe-key, иначе одинаковые
   пересборки никогда не совпадут. Persistent-cache/service не создавать.
9. `.claude/skills/audit-pack-generator/scripts/create_audit_pack.py` больше не
   содержит второй builder: оставить совместимый wrapper на канонический
   `scripts/make_audit_pack.py`; skill-команду переключить на канон.
10. До записи manifest выполнить fail-closed scan всех файлов пакета на
    присваивания ключей с суффиксами `_TOKEN`, `_SECRET`, `_API_KEY`, маркеры
    `Bearer`, `sk-` и bot-token-подобные строки. Находка блокирует пакет.
11. Для code-ТЗ `preflight.py` требует `--claude-receipt`; канонический receipt
    создаётся только штатным read-only Claude-вызовом и проверяет
    manifest/prompt/files/HEAD. Он является process gate, но как локальный
    unsigned-файл не доказывает запуск против процесса с правом переписать сам
    verifier. До валидного receipt код запрещён.

## СТОП

- требуется новый общий pack-builder;
- пакет читает секреты или клиентские данные;
- manifest пишется не последним;
- Claude получает Write/Edit/live tools;
- отдельный `.claude` builder продолжает писать пакет самостоятельно;
- требуется новый persistent state или daemon;
- бюджет более 150 строк нетестового кода.

## Приёмка

1. Тест-команда зелёная.
2. SHA-проверка проходит на неизменном пакете и падает после изменения байта.
3. Тестовый телефон и email маскируются.
4. Тестовый env/token-файл не читается и не включается; не заявлять общий
   secret-redaction, пока такого маскера нет.
5. Токен, помещённый прямо в любой входной файл пакета, блокируется до manifest.
6. Старый `.claude` entrypoint создаёт канонический пакет через wrapper.
7. Read-only Claude CLI читает явный пакет, называет HEAD и не меняет Git.
8. Новых зависимостей, сервисов и runtime/feature-флагов нет; новые CLI-аргументы
   являются явной частью этого ТЗ.
