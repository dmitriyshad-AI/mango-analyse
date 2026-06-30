> DONE 2026-06-30 19:40 | ветка codex/tz156-price-valid-until | codex

> TAKE 2026-06-30 19:23 | ветка codex/tz156-price-valid-until | codex

Ветка: codex/tz156-price-valid-until
Зоны: product_data/knowledge_base/, scripts/, tests/, tasks/, audits/_inbox/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q
Семантический-аудит: да

# ТЗ-156 — Продлить valid_until 12 ценовых фактов до 2026-12-31 (цены стабильны, бизнес-изменений нет)

- Дата: 2026-06-30. Постановщик: Дмитрий (цены актуальны, с 1 июля изменений НЕТ). Исполнитель: **D6** (или D1). Регрейд — мой. Канон: main; снапшот пилота `kb_release_20260612_v6_7_staging_r4_1` (пин в dialogue_contract_pipeline.py:56).
- Это правка ДАННЫХ (метка срока), НЕ изменение цен. Числа цен и бренды НЕ трогать.

## Зачем
Живой бот соблюдает `valid_until` (`_direct_path_valid_until_ok`, support.py:613). У 12 ценовых фактов метка `valid_until=2026-07-01`/`2026-08-01` → бот выкинет ещё действующую цену со 2 июля / 2 августа. Цена не меняется → продлеваем метку до `2026-12-31`.

## ⚠ Ловушка пересборки (главное — иначе правка протухнет)
`valid_until` ВЫВОДИТСЯ из ключа `before_2026_07_01` при сборке (`valid_until_from_path`, build_kb_release_v3_from_claude_handoff.py:1857,2466). Просто отредактировать снапшот НЕЛЬЗЯ — пересборка вернёт 1 июля. Но **явный `valid_until` в исходном item/structured имеет приоритет** (build_kb_release_v3:1855: `item.get("valid_until") or structured.get("valid_until") or valid_until_from_path`). Значит чиним в ИСТОЧНИКЕ.

## Что сделать
1. В ИСТОЧНИКЕ базы (handoff-данные, которые читает build_kb_release_v3) для этих 12 фактов проставить **явный `valid_until: "2026-12-31"`** (в item и/или structured_value — туда, что имеет приоритет), чтобы перекрыть вывод из ключа и пережить пересборку. Ключи (`before_2026_07_01`) НЕ переименовывать (используются в коде; имя станет косметически устаревшим — это в бэклог, не сейчас).
2. **Клиентский текст:** проверить `client_safe_text` этих 12 фактов на оговорку срока («действительна до 1 июля» и т.п.; см. include_price_validity build_kb_release_v3:2388). Раз цена стабильна — убрать/нейтрализовать дату-дедлайн из клиентского текста (не обещать клиенту «до 1 июля»). Сами числа цены не менять.
3. Пересобрать пиннутый релиз (`v6_7_staging_r4_1`) или выпустить корректного преемника и убедиться, что **пин пилота отдаёт исправленные факты**. Путь воспроизводимый (через билдер, не правкой JSON руками в обход источника).

## Список 12 фактов (точный)
foton: `prices_regular_2026_27.offline_3_4_class.before_2026_07_01.{semester,year}`, `.offline_5_11_class.before_2026_07_01.{semester,year}`, `.online_3_4_class.before_2026_08_01.{semester,year_range}`, `.online_5_11_class.before_2026_08_01.{semester,year}`
unpk: `prices_regular_2026_27.offline_1_4_class.before_2026_07_01.{semester,year}`, `.offline_5_11_class.before_2026_07_01.{semester,year}`

## Приёмка (моя по сырью)
- В отдаваемом пилотом снапшоте у всех 12 фактов `valid_until = 2026-12-31`.
- **Числа цен НЕ изменились** (дифф значений = 0), бренды целы, `allowed_for_client_answer=True` сохранён.
- В `client_safe_text` нет дедлайна «до 1 июля/августа».
- Пересборка из источника воспроизводит `valid_until=2026-12-31` (правка не выводится обратно из ключа).
- Полный pytest зелёный. Живой бот/CRM не трогать (только данные/выпуск базы).

## Файлы
исходник handoff базы (вход build_kb_release_v3), `scripts/build_kb_release_v3_from_claude_handoff.py` (1855-1859, 2388, 2466 — логика valid_until/текста), снапшот `kb_release_20260612_v6_7_staging_r4_1/kb_release_v3_snapshot.json`.
