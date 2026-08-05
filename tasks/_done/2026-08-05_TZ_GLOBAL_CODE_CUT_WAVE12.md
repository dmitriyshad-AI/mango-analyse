> DONE 2026-08-05 15:37 | ветка codex/global-code-cut-wave12 | codex

> TAKE 2026-08-05 15:29 | ветка codex/global-code-cut-wave12 | codex

Ветка: codex/global-code-cut-wave12
Зоны: .env.example, scripts/, src/mango_mvp/customer_profile/, tests/, docs/, tasks/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_customer_profile_builder.py
Семантический-аудит: да

# Уборка wave 12: старый CRM summary и флаг phone index

## Проблема

1. Старый `customer_profile.crm_summary` вызывается только своим preview CLI и
   тестами. Живые bot/AMO/Timeline/manager-dossier пути его не импортируют.
2. `build_insight_readiness_from_canonical.py` является одноразовым скриптом мая
   и динамически загружает уже удалённый `build_post_backfill_amo_ready_export.py`.
3. `PROFILE_PHONE_INDEX` с июня включён по умолчанию; выключенное состояние не
   используется ни одной текущей конфигурацией, но держит две схемы записи и
   тесты старого CRM preview.

## Образ результата и бизнес-польза

- Удалён невызванный старый слой CRM-текста и сломанный одноразовый скрипт.
- Profile phone index становится единственным простым путём без флага и второй
  INSERT-ветки; старые БД мигрируют штатным `_ensure_phone_index`.
- Живой AI-employee draft, Customer Timeline, AMO read-only и manager dossier не
  меняются.
- Удаляется больше 1100 строк; добавляется только один тест неизменяемого
  инварианта индекса.

## Минимальное решение

1. Удалить четыре невызванных файла старого контура.
2. Перенести только полезный контракт индекса в
   `tests/test_customer_profile_builder.py`.
3. Убрать `PROFILE_PHONE_INDEX` и всегда использовать уже существующий индекс;
   не создавать новый helper, фасад или флаг.

## Приёмка

1. Graphify и raw grep не находят живых импортов удаляемого CRM summary/readiness.
2. `PROFILE_PHONE_INDEX` не остаётся в runtime/config/tests; профильная таблица
   всегда содержит `primary_phone_norm` и индекс, значение нормализовано.
3. Profile tests, collect-only и полный pytest не дают новых падений.
4. Сквозной bot/Timeline тест подтверждает, что живой manager context не зависит
   от удалённого модуля.
5. Новых файлов runtime, флагов и зависимостей — ноль; общий diff отрицательный.

## Ограничения

- Не удалять `customer_profile` builder/store/contracts целиком.
- Не менять Customer Timeline, manager dossier или bot runtime.
- Не заменять удалённый preview новым текстовым генератором.

## СТОП

- Найден живой импорт, runtime config или active task удаляемого модуля.
- Always-on индекс ломает открытие/миграцию существующей профильной БД.
- Для сохранения поведения нужен новый слой сопоставимого размера.
