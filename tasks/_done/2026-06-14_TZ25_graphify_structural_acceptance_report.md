# TZ-25 Graphify Structural Layer — отчёт по приёмке

Дата: 2026-06-14
Worktree: `/Users/dmitrijfabarisov/Projects/Mango_graphify_tz25`
Ветка: `codex/tz25-graphify-structural`

## Объём

Сделан только структурный слой Graphify для локальной навигации по репозиторию. Облачный смысловой слой не запускался. `graphify-out` не коммитится.

Карта — только подсказка для навигации. Факт, число, цитату промпта, P0, разделение брендов и гварды нужно подтверждать в сыром исходнике. Для P0/брендов/гвардов первый источник — `src/mango_mvp/channels/rules_engine.py`.

## Gate A — pin пакета

PASS.

- Пакет: `graphifyy`
- Версия: `graphify 0.8.39`
- Требуемый commit: `fd470faeee16e9f42e3f47204824a2002a1f899c`
- `direct_url.json` содержит этот commit.
- Проверенные источники: официальный GitHub `https://github.com/safishamsi/graphify`, metadata установленного пакета, README на pinned commit.

## Gate B — структурная сборка

PASS.

- Источник сборки: чистый `git archive`, не рабочее дерево.
- Ревизия сборки: `7b3365ae15e703ea926ae0f3ac081521c0831128`
- Output вне git: `/Users/dmitrijfabarisov/Projects/Mango_graphify_tz25_graphify_structural/output/graphify-out`
- В output только: `GRAPH_REPORT.md`, `graph.json`, `mango_structural_manifest.json`
- Скопировано code-файлов из `src/**`: 304
- Graphify extractor увидел: 305 code files, 0 docs, 0 papers, 0 images
- Структурированных файлов KB v6.5 внесено как generated Python metadata: 30
- Узлов-ярлыков для raw dumps: 1, policy `label_only_no_content`
- Граф: 8841 nodes, 34444 edges, 0 hyperedges
- Token cost в `GRAPH_REPORT.md`: input=0, output=0

Воспроизводимость: PASS. Две сборки дали одинаковые хеши.

- `GRAPH_REPORT.md`: `181e45049e03563c2da80790ef11e127550d1fcb48d9e7449bfa34a9ab408df3`
- `graph.json`: `f40c7bef6f593d299739d8a79c7fca6834afb58fbd3affbc5e49298c15e83412`
- `mango_structural_manifest.json`: `b0ec3398f99f800ead5d3c778fbb61d8aad57fe7fd3776ce88c748cfb1dbd5a3`

## Gate C — cloud semantic

NOT_RUN_BY_DESIGN.

Реализация удаляет LLM env-ключи для build/query subprocesses и использует `graphify extract --no-cluster` по code-only input. Смысловая обработка прозы отложена по ТЗ.

## Gate D — локальный read-only server

PASS.

Встроенный `graphify-mcp` показал PR/GitHub-oriented tools, поэтому он не используется как проектный server. Добавлен `scripts/graphify_structural_mcp_stdio.py`: минимальный stdio JSON-RPC server только с локальными read-only tools:

- `query_graph`
- `get_node`
- `get_neighbors`
- `graph_stats`
- `shortest_path`

HTTP server, write tools и command execution отсутствуют.

## Gate E — контрольная архитектура до сборки

PASS.

`ARCHITECTURE.md` был закоммичен до первой сборки графа:

- Commit: `c85e711 Add Graphify control architecture map`
- 12 структурных строк с нетривиальными связями P0/брендов/гвардов и правилом “карта для навигации, истина в сырье”.

## Правило AGENTS.md

PASS. В `AGENTS.md` добавлен раздел Graphify:

- Карта для навигации, сырьё — источник истины.
- Отсутствие узла на stale-карте не доказывает отсутствие в коде.
- P0/brand/guard claims проверять в `src/mango_mvp/channels/rules_engine.py`.
- `graphify-out/` — локальный чувствительный output, не коммитить и не передавать наружу.
- Только stdio read-only; HTTP/write запрещены.
- Wrapper возвращает source path hints и revision banner.

## Пилот

PASS.

Вопросы были письменно зафиксированы до пилота:

- Main set: 15 вопросов
- Counter set: 5 вопросов
- Файл: `D1_audit_backlog/GRAPHIFY_TZ25_pilot_questions_2026-06-14.md`

Финальный пилот по пересобранному графу:

- Total: 20
- Command failures: 0
- Missing revision banner: 0
- Missing source path hint: 0
- Counter questions without guard note: 0

## Safety checks

PASS.

Скан финального `graphify-out`:

- Phone-like hits: 0
- Email hits: 0
- Внутренние маркеры сохранены: `internal_only=32`, `manager_only_route=31`, `forbidden_for_client=31`
- Manager/internal ошибочно помечены как client-safe: 0

## Тесты

PASS.

- `python3 -m py_compile src/mango_mvp/graphify_structural.py scripts/graphify_structural_build.py scripts/graphify_structural_query.py scripts/graphify_structural_mcp_stdio.py`
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest --collect-only -q tests/test_graphify_structural.py -p no:cacheprovider` -> 7 collected
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_graphify_structural.py -p no:cacheprovider` -> 7 passed
- Focused zone: `tests/test_graphify_structural.py tests/test_rules_engine.py tests/test_bot_safety_detector.py tests/test_answer_safety_classifier.py tests/test_question_catalog_safety.py tests/test_project_inventory.py tests/test_tz12_pii_gitignore.py` -> 181 passed

## Примечание

`graphify-out` находится вне git. Если после сборки добавлен новый commit, wrapper покажет stale-map banner до повторной сборки:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/graphify_structural_build.py --summary /Users/dmitrijfabarisov/Projects/Mango_graphify_tz25_graphify_structural/structural_build_summary.json
```
