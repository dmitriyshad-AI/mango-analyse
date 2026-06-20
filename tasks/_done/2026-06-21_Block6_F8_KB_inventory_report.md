# Block6 F8 KB inventory — night run

Дата: 2026-06-20/21
Режим: read-only, код/данные KB не менялись.

## Snapshot

Актуальный KB snapshot:

`/Users/dmitrijfabarisov/Projects/Mango analyse/product_data/knowledge_base/kb_release_20260612_v6_7_staging_r4_1/kb_release_v3_snapshot.json`

Причина выбора: r4.1 указан как текущий пилотный снимок в проектном состоянии и runtime-дефолтах.

## Сводка

- `run_id`: `kb_release_20260612_v6_7_staging_r4_1`
- `generated_at`: `2026-06-12T08:13:44.975221+00:00`
- фактов: `1075`

## structured_value

- `amount/price` как термин внутри `structured_value`: `76` фактов.
  - `amount`: `55`
  - `price`: `59`
  - пересечение `amount ∩ price`: `38`
- Только явные верхние поля: `58` фактов.
- `tariff` внутри `structured_value`: `14`.
- Явного верхнего поля `tariff` нет; найдено `forbidden_old_tariff_name`: `1`.
- `subject`: ожидание ТЗ `0` не подтвердилось.
  - явное верхнее поле `subject`: `2`
  - как термин внутри `structured_value`: `40`
- `grade`: ожидание «только в тексте» не подтвердилось.
  - явное верхнее поле `grade`: `2`
  - вложенное `applies_to.grades`: `1`
  - как термин внутри `structured_value`: `6`

## amount/price и числа в client_safe_text

По широкому критерию `amount/price` внутри `structured_value`:

- всего: `76`
- с цифрой в `client_safe_text`: `48`
- без цифры: `28`
- из них `24` имеют `allowed_for_client_answer=false` и пустой `client_safe_text`.

Клиентски разрешённые факты с amount/price, но без цифры в `client_safe_text`:

- `fact:v3:foton:processes_2026_06_10_foton_payment_semester_year_price:9c3e0d1b92`
- `fact:v3:foton:lvsh_mendeleevo_2026_transfer_from_moscow_included_in_price:98f7a84078`
- `fact:v3:unpk:processes_2026_06_10_unpk_payment_semester_year_price:41cb9bd31d`
- `fact:v3:unpk:prices_regular_2026_27_online_olympiad_phystech_classes_client_safe_text:d4c4718a9d`

## Вывод для будущего ТЗ

Оси `amount/price` и `tariff` в целом совпали с ожиданием ночного ТЗ. Оси `subject` и `grade` уже частично структурированы в r4.1, поэтому будущее ТЗ «оси подбора продукт+цена» должно учитывать реальные поля, а не исходить из `subject=0` и `grade only text`.
