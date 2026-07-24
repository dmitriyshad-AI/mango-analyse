> DONE 2026-07-24 12:52 | ветка codex/kb-podlipki-20260724 | codex

> TAKE 2026-07-24 12:20 | ветка codex/kb-podlipki-20260724 | codex

Ветка: codex/kb-podlipki-20260724
Зоны: product_data/knowledge_base/, scripts/build_kb_release_v3_from_claude_handoff.py, scripts/build_kb_release_v6_1_team_answers.py, src/mango_mvp/channels/fact_venue_scope.py, src/mango_mvp/channels/fact_scope_spec.py, src/mango_mvp/channels/conversation_intent_plan.py, src/mango_mvp/channels/subscription_llm_parts/direct_path.py, src/mango_mvp/knowledge_base/product_existence_axes_catalog.py, tests/, tasks/, docs/worktrees_registry.md
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_kb_release_v3_import.py tests/test_kb_distribution_packs.py
Семантический-аудит: да

# ТЗ: новая смена УНПК МФТИ «Подлипки» в базе знаний

Источник владельца: Google Doc `1bMhN0DtqNK8Z2XdwGMci2lAv0CtSYQ4QGb1Hr4dQ9Oo`, вкладка `t.gnwj6o2a1spz`.

1. Добавить отдельный воспроизводимый YAML-источник смены «Подлипки» и зарегистрировать его в release manifest.
2. Добавить клиентские факты только бренда УНПК: место, даты, формат, классы, физмат, проживание, питание, трансфер, медблок, вместимость, цена 130 000 ₽, открытый набор.
3. Разрешить противоречие источника безопасно: 5–10 классы заявлены, группа 10 класса подтверждается после достаточного набора.
4. Оставить manager-only: минимум 114 000 ₽, скидки 5/10/15%, точные детали трансфера, подтверждение конкретной группы.
5. Обновить общий процесс ЛВШ УНПК так, чтобы распроданное Менделеево не скрывало открытую смену «Подлипки».
6. Сохранить площадочные метки при штатной пересборке и выделить `lvsh_podlipki` отдельно от `lvsh_mendeleevo`.
7. Пересобрать канонический snapshot и distribution packs штатным builder, проверить формальные гейты и отдельный смысловой аудит.

Запрещено: live-write, AMO/Wappi/CRM, раскрытие внутренних цен и скидок, смешение с Фотоном.

## Приёмка

- новая смена воспроизводится из YAML-источника штатной сборкой;
- клиент видит только 130 000 ₽, внутренние 114 000 ₽ и скидки отсутствуют в client-safe pack;
- Подлипки не смешиваются с Менделеево и Фотоном;
- формальные тесты и отдельный смысловой аудит зелёные.

## СТОП

- расхождение с документом владельца, утечка внутренних условий или чужого бренда;
- штатная пересборка удаляет существующие площадочные метки либо меняет несвязанные факты;
- любой live-write или изменение runtime-контура.
