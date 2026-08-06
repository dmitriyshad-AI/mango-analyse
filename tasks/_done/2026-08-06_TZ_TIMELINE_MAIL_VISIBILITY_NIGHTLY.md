> DONE 2026-08-06 03:19 | ветка codex/timeline-stage4b-nightly-20260806 | codex

> TAKE 2026-08-06 02:08 | ветка codex/timeline-stage4b-nightly-20260806 | codex

Ветка: codex/timeline-stage4b-nightly-20260806
Зоны: src/mango_mvp/customer_timeline/mail_link_enrich.py, src/mango_mvp/customer_timeline/stage4b_bot_opening.py, src/mango_mvp/customer_timeline/nightly_service.py, src/mango_mvp/customer_timeline/bot_safe_runtime_context.py, scripts/build_customer_timeline_nightly_dv2_sources.py, scripts/run_customer_timeline_codex_task.py, scripts/run_customer_timeline_nightly_service.py, scripts/publish_snapshot/reader_smoke.py, tests/test_customer_timeline_mail_link_enrich.py, tests/test_customer_timeline_stage4b_bot_opening.py, tests/test_customer_timeline_nightly_service.py, tests/test_customer_timeline_codex_task.py, tests/test_bot_safe_runtime_context.py, tests/test_publish_snapshot_tooling.py, docs/worktrees_registry.md, docs/MANGO_CURRENT_STATE_V7.md, docs/DECISIONS_LOG.md, tasks/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_customer_timeline_mail_link_enrich.py tests/test_customer_timeline_stage4b_bot_opening.py tests/test_customer_timeline_nightly_service.py tests/test_customer_timeline_codex_task.py tests/test_bot_safe_runtime_context.py tests/test_publish_snapshot_tooling.py
Семантический-аудит: да

# ТЗ: сохранить принятую видимость почты и сделать Stage4b частью nightly

## Проблема

`mail_link_enrich` обязан обновлять identity-связь и по D-060 не менять уже
принятое решение о видимости. Сейчас UPDATE безусловно ставит
`a2v3_mail_event_facts.bot_visible=0`, поэтому последующий Stage4b не может
открыть ни одного письма. Сам Stage4b существует, но nightly его не вызывает.

## Образ результата и бизнес-польза

Черновик менеджеру получает только письма точно связанного клиента, которые уже
прошли A2-проверку. Повторный enrich не закрывает разрешённое письмо, а ночная
сборка каждый раз пересматривает разрешения и закрывает устаревшие. Чужое письмо
никогда не попадает в контекст. Публикация невозможна при нарушении гейта.

## Минимальное решение

1. При UPDATE существующего A2-факта сохранять `bot_visible` и его текущую
   причину. Новая строка по-прежнему создаётся закрытой.
2. Подключить существующий `run_stage4b_bot_opening` обязательным шагом после
   family/derived refresh и перед bot-safe rebuild. Новых модулей и флагов нет.
3. Не ослаблять D-076: почта открывается только при A2 `bot_visible=1`, точной
   связи и отсутствии запрещённых тегов.
4. Не выполнять второй полный quick-check внутри Stage4b, если итоговый
   quick-check делает nightly; промежуточные нарушения остаются fail-closed.
5. Последний PII-гейт не ослаблять: адрес в разрешённом письме заменять уже
   существующей санитаризацией, а не выбрасывать весь полезный фрагмент.

## Рассмотрено и отвергнуто

- Ослабить Stage4b-гейт: нарушает D-076.
- Переписать видимость в bot-safe rebuild: смешивает identity и публикационную
  политику, дублирует существующий Stage4b.
- Новый восстановительный скрипт: сначала переиспользовать существующие A2
  факты/снимки и штатный runner.

## Приёмка

1. Регрессия: существующий `bot_visible=1` остаётся 1 после enrich; новая запись
   остаётся 0; изменение связи на другого клиента не переносит старое разрешение
   без повторной A2-проверки.
2. Сквозной факт: письмо точно связанного клиента доходит до фактического текста
   `build_bot_safe_crm_context`; письмо чужого клиента не доходит.
3. Лестница на APFS-клоне: 1 → 10 → 100 → весь массив. Перед каждой ступенью
   записаны ожидания; после — баланс, отрицательный контроль и повтор без дублей.
4. Stage4b присутствует в конфиге и обязательной цепочке. Его FAIL/нарушение
   делает nightly partial/failed и не меняет `latest_published`.
5. Один итоговый `PRAGMA quick_check`, FK=0; COUNT первичных событий до=после.
6. Формальный, data, breaker и business/semantic verdict указаны раздельно.
7. БД `product_data` и внешние AMO/Tallanto/Wappi не изменяются.

## СТОП

- Любая попытка открыть письмо без A2 `bot_visible=1` или точной связи.
- Любое изменение числа первичных событий, чужой customer_id в контексте,
  плохой quick-check/FK или смена `latest_published` после partial/failed.
- Отсутствие локального проверяемого входа для восстановления прежних A2-флагов:
  не угадывать разрешения и не ослаблять гейт.
- Неожиданная грязь вне зон ТЗ или необходимость писать в live/product_data.

## Бритва

Бюджет нового интеграционного механизма: до 150 добавленных строк нетестового
кода. Новых файлов кода и зависимостей: 0; один execution-control для передачи
единственного quick-check от Stage4b ночной службе. Удаление/упрощение
приветствуется, но только внутри зон ТЗ.
