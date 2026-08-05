> DONE 2026-08-05 17:43 | ветка codex/timeline-final-data-gates | codex

> TAKE 2026-08-05 16:11 | ветка codex/timeline-final-data-gates | codex

Ветка: codex/timeline-final-data-gates
Зоны: src/mango_mvp/customer_timeline/, tests/, docs/, tasks/, audits/_inbox/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_customer_timeline_family_graph.py tests/test_wappi_history_import_to_timeline.py tests/test_wappi_history_checkpoint.py tests/test_customer_timeline_tallanto_attendance_import.py tests/test_customer_timeline_nightly_service.py
Семантический-аудит: да

# Финальные data-гейты Customer Timeline

## Проблема

Свежая staging-витрина остаётся partial. Причины разделены:

1. family graph не признаёт позднюю точную `identity_links` у старой карточки
   `match_status=inferred`; 229 событий 34 клиентов теряют ребёнка;
2. Wappi смешивает pre-guard, post-guard и успешно сохранённые старые привязки:
   176 resolved выглядят pending, 141 групповой message считается клиентским;
3. ночь исполняется из старой ветки, поэтому 14 из 16 посещений не используют
   уже влитый exact Tallanto resolver.

## Образ результата

- точный уникальный Tallanto ID того же клиента признаётся один раз общим
  авторитетом; чужой/неуникальный/открыто конфликтный ID остаётся закрытым;
- Wappi имеет один замкнутый post-guard баланс: linked + quarantine + junk;
- явный сохранённый quarantine не выдаётся за потерю источника и не замораживает
  всю витрину, но `attribution_complete` остаётся false до разбора;
- night service выполняется из зафиксированного current SHA; никакого prod/live
  write и автоматической публикации нет.

## Минимальный вариант

Новый resolver/queue отклонён как дубль. Полное отключение гейтов отклонено.
Переиспользуются `authoritative_exact_identity_rows`, существующий Wappi guard,
`timeline_conflicts` и текущий attendance resolver.

## Приёмка

- family graph: поздняя exact link восстанавливает безопасное событие; чужой ID,
  duplicate owner, конфликт и подозрительное имя не открываются;
- на clone `tallanto_student_id_not_in_family` 495 -> 281, второй проход идентичен;
- Wappi invariant замкнут и отдельно показывает linked/quarantine/junk;
- `pending_reason_counts` не содержит resolved;
- старые attendance-отношения переоценены: unmatched=0, один точный conflict остаётся;
- targeted/full tests без новых падений, quick_check=ok, FK=0;
- новых LLM-вызовов, resolver, очередей, флагов и зависимостей нет.

## СТОП

- exact ID имеет другого/нескольких владельцев либо адресный открытый конфликт;
- для replay нужен новый сетевой или write-механизм;
- изменение требует prod DB, AMO/CRM/Tallanto/Wappi write или client send;
- баланс не замыкается на одном знаменателе.

## Фактический результат

- Family graph: `495 -> 281`; восстановлено 214 безопасных событий. Остаток не
  открыт автоматически: подозрительные имена, прямой exact-ID conflict, нет
  exact-связи или exact принадлежит другому владельцу. Два прохода идентичны.
- Attendance на свежем staging-клоне: 1 514 отношений разрешены, 541 событие
  создано, 1 точный конфликт сохранён, 85 legacy-unmatched конфликтов
  переоценены и закрыты. Повтор: 0 created/updated, 15 duplicate.
- Wappi: сетевой баланс отделён от локальной перепривязки; явный карантин не
  выдаётся за потерю источника и не блокирует публикационный гейт.
- Прямых внешних write, prod-write, client-send и новых LLM-вызовов нет.
