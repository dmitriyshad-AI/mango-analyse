> DONE 2026-07-02 03:46 | ветка codex/adr003-semanticframe-migration | codex

> TAKE 2026-07-02 03:38 | ветка codex/adr003-semanticframe-migration | codex

Ветка: codex/adr003-semanticframe-migration
Зоны: scripts/, tests/, tasks/, audits/_inbox/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_report_adr003_existence_fact_verification.py tests/test_report_adr003_overhandoff_levers.py
Семантический-аудит: да

# TZ ADR-003 F2c: existence/format fact-verification shadow

## Контекст

F2/F2b показали: быстрый route-only active не готов. На свежем M1-прогоне `36ea110` безопасные ответы не дают чистых кандидатов на понижение маршрута:

- `would_demote=0`;
- `harmless_context_ack_status_candidates=0`;
- основная ошибка SemanticFrame в gold — он путает справочный вопрос "существует ли курс/формат/группа для класса X" с live availability / enroll.

При этом активный ответ на такие вопросы нельзя включать без доказанного факта: иначе получим выдумку про существование курса, группы, класса или формата.

## Цель

Сделать read-only отчёт F2c: для проблемных ходов из F2b проверить, есть ли в runtime-транскрипте и/или KB snapshot свежий client-safe факт, который подтверждает существование/формат запрошенного продукта.

Это только диагностический shadow-слой. Поведение бота не меняется.

## Scope

1. Новый скрипт `scripts/report_adr003_existence_fact_verification.py`.
2. Тесты `tests/test_report_adr003_existence_fact_verification.py`.
3. Прогон на сырье последнего F2b/36ea110:
   - ON transcripts;
   - gold calibration/report;
   - KB snapshot `product_data/knowledge_base/kb_release_20260612_v6_7_staging_r4_1/kb_release_v3_snapshot.json`.
4. Audit pack в `audits/_inbox/adr003_f2c_existence_fact_verification_<timestamp>/`.

## Инварианты

- Не менять runtime-код бота, route/text, direct path, профиль, live, P0-floor/preblock.
- Не добавлять новый regex runtime-детектор.
- Любые текстовые/маркерные проверки в этом шаге допустимы только как offline scorer/report, не как боевое правило.
- Ничего не писать в AMO/Tallanto/CRM/Telegram.
- Не трогать live/Wappi.
- Не трогать мораторий-файлы ADR-003.

## Acceptance

- Скрипт строит JSON + Markdown report.
- Report явно делит:
  - already self;
  - current handoff without fact proof;
  - handoff with KB evidence candidate;
  - no evidence / needs KB/retriever work;
  - danger-adjacent / money / P0 excluded.
- Report показывает, можно ли следующий активный шаг строить как route-only или нужен fact-verification/retriever layer.
- Тесты покрывают exact KB evidence, partial/no evidence, wrong brand, stale/not client-safe fact.
- `pytest` по новым и F2b тестам зелёный.
- Semantic review фиксирует, что это formal+semantic diagnostic only, не разрешение включать активный self-answer.
