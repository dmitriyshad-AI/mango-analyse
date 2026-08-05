> DONE 2026-08-05 16:06 | ветка codex/global-code-cut-wave13 | codex

> TAKE 2026-08-05 16:00 | ветка codex/global-code-cut-wave13 | codex

Ветка: codex/global-code-cut-wave13
Зоны: scripts/, src/mango_mvp/question_catalog/, tests/, docs/, tasks/, audits/_inbox/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_question_catalog_deal_aware_bridge.py tests/test_question_catalog_classifier_v2.py tests/test_build_adr003_semantic_frame_eval.py tests/test_report_adr003_semantic_frame_eval.py
Семантический-аудит: нет

# Уборка wave 13: закрытые вопросные и ADR-калибровочные контуры

## Проблема

После удаления deal-aware и старых M1-контуров остались одноразовые июньские
сканеры вопросов, review-обвязка Question Catalog и завершённая frame-gold
калибровка ADR-003. Они не вызываются runtime, службами, текущими задачами или
живыми тестами, но поддерживают ложное впечатление нескольких активных путей.

## Образ результата

Удалены только доказанно закрытые владельцы. Живые classifier/extractors,
текущий ADR/P0 измеритель и Stage 15 остаются. Импорт удалённого модуля падает,
а используемый извлекатель вопросов и текущий model-led измеритель работают.

## Приёмка

- Graphify и сырой поиск не находят production-caller удаляемых модулей;
- живые Question Catalog и ADR/P0 наборы зелёные;
- полный collect-only и полный pytest не получают новых падений;
- runtime-дифф отрицательный; новых файлов кода, флагов и зависимостей нет.

## СТОП

- найден production-caller, служба или активное ТЗ, использующее кандидата;
- удаление требует нового runtime-механизма или меняет клиентское поведение;
- текущий P0-измеритель либо живой Question Catalog перестаёт собираться.
