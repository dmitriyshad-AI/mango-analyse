> DONE 2026-07-20 04:05 | ветка main | codex

> TAKE 2026-07-20 03:50 | ветка main | codex

Ветка: main
Зоны: src/mango_mvp/replay_exam/pseudonymizer.py, tests/test_wappi_replay_pii_scan.py, scripts/publish_snapshot/config.marathon2_noch_current.json, tests/test_publish_snapshot_tooling.py, docs/RUNBOOK.md, tasks/, audits/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_wappi_replay_pii_scan.py tests/test_publish_snapshot_tooling.py tests/test_subscription_llm_draft_provider.py
Семантический-аудит: нет

# Дальнейшие действия после 6e232ffe

Источник: `/Users/dmitrijfabarisov/Claude Projects/Foton/2026-07-20_TZ_dalneyshie_deystviya.md`.

## Цель

Закрыть безопасные операционные блокеры без изменения поведения живого бота:

1. Опубликовать уже проверенный `6e232ffe` в `origin` и `yandex`.
2. Убрать ложные PII-срабатывания только для точных публичных URL из контактных KB-фактов и валидных generated hex-хешей. Клиентские ПДн не разрешать.
3. Обновить publish-конфиг на канонический worktree и фактическую staging-БД; пересчитать пять контрольных baseline в read-only и получить зелёный `reader_smoke` без build/flip/rollback.
4. Подтвердить существующий downgrade-only тест authoritative gate, не дублировать его.
5. Дополнить RUNBOOK чек-листом будущего Wappi-редеплоя. Редеплой не выполнять.
6. После зелёных проверок обновить существующий M1-пакет только необходимой версией PII-gate и его хешами, не создавая второй пакет.

## NEG

- Не менять direct-path, P0, бренд, факты, маршрутизацию или клиентский текст.
- Не выполнять Customer Timeline build/flip/rollback и не писать в staging/prod.
- Не перезапускать Wappi, Telegram, calls или nightly.
- Не разрешать реальные телефоны, email, ФИО и произвольные значения в hash-полях.
- Не создавать архивы и копии мусора.

## СТОП

- Любая находка после калибровки, кроме подтверждённых публичных KB-контактов или валидного generated hex-хеша.
- Красный `reader_smoke`, `quick_check`, mail/mango safety gate или расхождение наблюдаемых счётчиков.
- Необходимость перезапуска службы, публикации БД либо записи во внешнюю систему.

## Приёмка

- `origin/main` и `yandex/main` содержат `6e232ffe`.
- Повторный скан сохранённого M1 FAIL даёт 0 находок; синтетический клиентский телефон и URL вне KB по-прежнему блокируются.
- Publish `reader_smoke` на текущей staging-БД зелёный, конфиг содержит наблюдаемые read-only счётчики.
- Существующий authoritative-gate тест зелёный.
- RUNBOOK описывает render-only, installer, `live_truth`, smoke, `exit 78` и возврат; live не менялся.
