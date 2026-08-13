> DONE 2026-08-13 11:15 | ветка codex/final-cleanup-regex-20260812 | codex

> TAKE 2026-08-12 20:23 | ветка codex/final-cleanup-regex-20260812 | codex

Ветка: codex/final-cleanup-regex-20260812
Зоны: src/mango_mvp/channels/, src/mango_mvp/pilot_context_assembly.py, scripts/, tests/, docs/, tasks/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_adr003_regex_understanding_moratorium.py tests/test_subscription_llm_draft_provider.py tests/test_draft_loop.py
Семантический-аудит: да

# ТЗ: завершить смысловой regex-аудит и удалить доказанный мёртвый код

## Проблема

Две волны уже удалили повторные смысловые слои, но frozen inventory всё ещё
содержит 701 текстовую проверку, из которых 412 не классифицированы. В живом
Wappi-пути до основной модели также собираются старые keyword-поля. Параллельно
в репозитории остались standalone-инструменты без живых entrypoint.

## Образ результата и бизнес-польза

- намерение, тема, P0, действие и подбор фактов в рабочем Wappi-черновике
  определяются существующими модельными полями, а не regex/keyword;
- P0/output/fact/identity/PII-полы остаются детерминированными проверками выхода;
- все оставшиеся текстовые проверки имеют явный класс и владельца;
- доказанно невызванные инструменты удалены без замены и без нового кода;
- Calls, Customer Timeline, runtime и внешние системы не меняются;
- строк рабочего кода становится меньше, новых флагов, зависимостей и LLM-вызовов нет.

## Минимальные варианты

1. Удалить только мёртвые инструменты и завершить классификацию — безопасный минимум.
2. Дополнительно убрать живой pre-model keyword-контур, если сырой call-chain и тест
   доказывают, что model-driven retriever и SemanticFrame уже являются владельцами.
3. Строить новый LLM-классификатор — запрещено как дублирование существующей модели.

Выбирается вариант 2 только для доказанного среза; спорные safety/fact-проверки
остаются с явным обоснованием.

## Приёмка

- Graphify на точном SHA используется как навигация, выводы подтверждены исходниками;
- рабочий Wappi entrypoint и конечная AMO-заметка проверены без внешних записей;
- новые/оставшиеся regex не решают intent/topic/P0/route/requested_action;
- удалённые файлы не имеют production callers, runtime/config refs и уникальной бизнес-функции;
- профильные тесты, moratorium и collect-only проходят;
- независимые architect, breaker, business и cleaner reviews выполнены;
- audit pack фиксирует до/после LOC, риски и то, что не менялось.

## Стоп-условия

- не трогать Calls/ASR, Customer Timeline, stable_runtime и внешние системы;
- не удалять P0/output/fact/identity/PII-полы;
- не обновлять snapshot ради сокрытия нового смыслового regex;
- не создавать fallback, feature flag или новый модельный вызов.
