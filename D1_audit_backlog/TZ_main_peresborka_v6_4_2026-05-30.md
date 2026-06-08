# ТЗ MAIN — легализация и пересборка v6.4 разрешённым скриптом. 2026-05-30.

Автор: Claude #1. Исполнитель: MAIN Кодекс (репозиторий). НЕ срочно — перед пилотом, НЕ блокирует
правку 4. Сборку/смысл фактов решает Дмитрий, Кодекс реализует.

## Зачем

Бот через `DEFAULT_KB_SNAPSHOT_PATH` читает снимок **v6.3** (`kb_release_20260520_v6_3_team_answers/
kb_release_v3_snapshot.json`). v6.4 (ветка `codex/kb-v6_4-20260529`, 4 коммита) в снимок НЕ пересобрана
и собрана ЗАПРЕЩЁННЫМ скриптом. Перед пилотом v6.4 надо легализовать и переключить бота на неё, иначе
клиент не увидит v6.4-факты (возврат после оплаты, цена УНПК 41 800/69 900, q15, expose client-safe).

## Что проверено по коду (Claude #1) — это НЕ «просто пересобрать»

Разрешённый `build_kb_release_v6_1_team_answers.py` читает `_sources/release_manifest.yaml` + facts
YAML; политика прямо в коде: «business overrides from release_manifest.yaml, not Python»
(`source_mutation_policy: no_business_patches_in_python`).

НО v6.4 (4 коммита) добавил бизнес-логику ПРЯМО в запрещённый `build_kb_release_v3_from_claude_handoff.py`:
- `REFUND_CLIENT_SAFE_POLICY_MARKERS` → `refund_post_payment` помечен client-safe;
- `enrich_phase2_structured_metadata()` — structured-метаданные фактов;
- изменён manual-факт `team_answers.q15.unpk_online_other_classes`: `fact_type` price→program, текст →
  «вне формата 2 раза/нед для 5-11 классов…», **`route_policy` manager_handoff_only → bot_answer_self_for_pilot**;
- правка `is_internal_child` для refund.

`release_manifest.yaml` в 4 коммитах НЕ менялся. Часть v6.4 (тексты refund, 41 800) в facts YAML есть,
но КЛАССИФИКАЦИЯ client-safe / route / metadata — только в Python запрещённого скрипта.

**Следствие:** чистый запуск `v6_1` НЕ воспроизведёт прогнанный Яндекс-артефакт v6.4 — потеряет
Python-логику (refund_post_payment может вернуться в internal; q15 — к manager_handoff). Значит сперва
перенос, потом сборка.

## Что сделать

1. Перенести бизнес-правки v6.4 из Python (`build_kb_release_v3_from_claude_handoff.py`) в правильный
   слой — `release_manifest.yaml` / facts YAML в `_sources`: классификация refund client-safe;
   structured-метаданные; факт q15 (тип/текст/route `bot_answer_self_for_pilot`). Перенос ≠
   переписывание: бизнес-смысл не менять, только сменить слой Python→YAML/manifest.
2. Собрать релиз разрешённым `scripts/build_kb_release_v6_1_team_answers.py` из обновлённых `_sources`.
3. **Сверить** новый снимок с прогнанным артефактом `_kb_v6_4_artifact_for_m1_20260529/.../kb_release_v3_snapshot.json`:
   refund_post_payment client-safe; q15 route=`bot_answer_self_for_pilot`; цена 41 800/69 900; expose
   client-safe факты — должны совпасть. Расхождения — разобрать.
4. Прогнать semantic gate `scripts/run_kb_semantic_review.py` на новом релизе (`semantic_pass=true`).
5. Переключить `DEFAULT_KB_SNAPSHOT_PATH` на новый v6.4-снимок.
6. Запрещённый скрипт: откатить v6.4-правки / пометить deprecated, чтобы не было двух источников правды.

## Негативный контроль (правило #4)

- internal-факты (юр.лица, лицензии) НЕ стали client-safe;
- разделение брендов не нарушено;
- `semantic_pass=true`, `quality_passed=true`, гейты целостности зелёные;
- число client-safe фактов правдоподобно (не схлопнулось/не раздулось против v6.3).

## Ограничения

- Сборку/semantic gate можно отдать Claude #1 на прогон в песочнике, если лимит main мал.
- Если перенос велик для одного захода — сообщить Дмитрию. Альтернатива (решает Дмитрий): зафиксировать
  как долг и пилотировать на v6.3 без v6.4-фактов — хуже по полноте, но без нарушения процесса.
- Отчёт: что перенесено (Python→YAML), результат сверки с артефактом, semantic_pass, что переключено.
