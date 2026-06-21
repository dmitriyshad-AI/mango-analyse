# D3 N1: phone/email в interest/title

Цель ревью: проверить, что N1 не требует правки кода, потому что в свежей тест-копии нет остаточных phone/email в `interest/title` выжимки.

Тестовая БД:

`/tmp/mango_phase01_integration_final_20260621_102959/customer_timeline.sqlite`

Проверенный слой:

- таблица `bot_context_chunks`;
- `source_system='customer_timeline_bot_safe_summary'`;
- поля `record_json.text`, `record_json.summary`, извлечённый фрагмент `Интерес: ...`.

Ключевой результат:

- всего выжимок: `18001`;
- видимых боту: `17901`;
- остаточные phone/email: `0`;
- рантайм-скан видимого текста: `0`;
- код не менялся.

Файлы пакета:

- `implementation_notes.md` — что проверялось и почему код не менялся;
- `scan_metrics.json` — машинные метрики скана;
- `test_output.txt` — результат целевых тестов;
- `semantic_review.md` — смысловой аудит;
- `risk_review.md` — остаточные риски;
- `backward_compatibility.md` — влияние на существующее поведение;
- `changed_files.txt` — список файлов, добавленных в отчёт.
