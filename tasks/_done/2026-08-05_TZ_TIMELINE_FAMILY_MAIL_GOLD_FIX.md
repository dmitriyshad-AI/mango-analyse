> DONE 2026-08-05 20:49 | ветка codex/timeline-family-mail-gold-fix | codex

> TAKE 2026-08-05 18:52 | ветка codex/timeline-family-mail-gold-fix | codex

Ветка: codex/timeline-family-mail-gold-fix
Зоны: src/mango_mvp/customer_timeline/family_graph.py, src/mango_mvp/customer_timeline/mail_link_enrich.py, tests/test_customer_timeline_family_graph.py, tests/test_customer_timeline_mail_link_enrich.py, docs/worktrees_registry.md, tasks/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_customer_timeline_family_graph.py tests/test_customer_timeline_mail_link_enrich.py
Семантический-аудит: да

# Timeline: семейная связь эталонной почты

## Бизнес-результат

Точные карточки разных детей одной семьи не остаются в разных `family_id` из-за
различающегося `parent_fio`. Почта получает семейную связь без назначения одному
ребёнку и без смешения двух разных семей.

## Доказанный исходный факт

- Все 155/155 писем ручного эталона физически есть в каноническом mail archive и
  в свежей staging Timeline.
- У контрольной семьи 39/39 писем загружены, но 0 связаны с exact student owner:
  16 `email_multiple_families_conflict`, 23 без внешнего адресата.
- Три exact Tallanto cards имеют разные структурные имена, общий телефон и email,
  но лежат в двух `family_id`: доказанное ядро из двух детей и один singleton.
- Широкое правило phone+email затронуло бы 514 разделённых групп. После ломающей
  проверки оно запрещено; безопасное расширение ядра затрагивает 2 карточки.

## Минимальное решение

В существующем `_resolve_family_assignments` разрешить строгое присоединение
одной exact Tallanto card к уже доказанному семейному ядру:

1. общий нормализованный телефон и email;
2. ровно одно ядро из минимум двух детей, уже объединённых штатным
   `parent_fio + contact`, и ровно один singleton;
3. по одному exact Tallanto student id на клиента, итоговая семья не больше 8;
4. структурная фамилия общая, имена и `student_type` различаются;
5. известные alias keys имён попарно не пересекаются;
6. несколько ядер и действующие `unsafe` запреты сохраняют раздельные семьи.

В `mail_link_enrich` убрать только избыточное требование
`customer_identity.identity_status == strong` для семейного кандидата. Итоговый
`family_strong` всё равно требует один persisted `family_id`, сохраняет
`customer_id = NULL` и не создаёт bot chunk.

Штатная переоценка конфликтов должна понимать старые `entity_refs` без
дублирующего resolver. Существующий ночной режим `reconsider_pending` повторно
проверяет уже сохранённые `family_strong`, чтобы новый конфликт снимал семейную
связь без массовой перепроверки индивидуальных `strong_unique`.

Переиспользовать существующие `union`, `normalized_name_tokens`,
`child_name_keys`, `_reconcile_contact_conflicts` и `mail_link_enrich`.

## Отвергнутые варианты

1. Править `mail_link_enrich`: второй семейный resolver и лечение симптома.
2. Удалить `parent_fio` из текущего ключа: слишком широкое слияние 514 групп.
3. Force-link письма одному ребёнку: нарушает границу чужого ребёнка.

## Приёмка

- Singleton exact card присоединяется к единственному доказанному ядру.
- Общий контакт двух разных семей остаётся разделённым.
- Дубль/вариант имени остаётся конфликтом.
- Письмо с именем ребёнка относится только к нему; письмо без имени не получает
  детскую атрибуцию.
- На копии staging: до/после по открытым конфликтам, 39 эталонным письмам,
  family IDs, quick_check/FK и идемпотентному второму проходу.

## Стоп

- Нужно писать в production/stable runtime или внешнюю систему.
- Правило создаёт семью с нуля, затрагивает больше двух singleton-карточек или
  допускает несколько семейных ядер.
- Появляется новый resolver, флаг, зависимость или отдельный pipeline.

## Результат 2026-08-05

- Контрольная семья: `2 -> 1` family_id.
- Эталонная почта: `blocked 16 / unmatched 23 -> family_strong 16 / unmatched 23`.
- Писем, назначенных конкретному ребёнку: `0`.
- Открытых конфликтов контрольной семьи: `6 -> 0`.
- Глобально `family_strong`: `76 -> 432`.
- Переоценено и закрыто `319` доказанно устаревших конфликтов, переоткрыто `0`.
- Второй проход: `0` обновлений, отпечаток совпал, `quick_check=ok`, FK-ошибок `0`.
- Timeline regression: `978 passed`.

Смысловой PASS относится к корректности семейной связи. Видимость этих писем в
промпте бота остаётся отдельным обязательным блоком.
