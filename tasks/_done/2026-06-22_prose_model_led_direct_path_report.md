# TELEGRAM_PROSE_MODEL_LED: direct-path prose quality

Дата: 2026-06-22  
Ветка: `codex/prose-model-led`  
База ветки: `codex/p0-model-led-complaint` @ `d949ccd`  
Флаг: `TELEGRAM_PROSE_MODEL_LED`, default OFF, в `pilot_gold_v1` не добавлен.

## Что изменено

- Direct-path prompt получил малый блок качества текста за флагом: модель пишет живой текст сама, не копирует служебные шаблоны, не обещает места, не выводит внутренние плейсхолдеры и не пишет клиенту мета-фразы вида «в фактах нет».
- Fallback `_promoted_verified_fact_text` под флагом убирает казённые обязательные зачины вроде «Да, сориентирую по проверенной информации/условиям».
- В `apply_authoritative_output_gate` после штатного output sanitizer и до authoritative findings добавлен `apply_prose_model_led_quality_guard`:
  - чистит `[данные у менеджера]`, `[...]`, `[…]` из клиентского текста;
  - убирает мета-фразы про отсутствие факта;
  - переписывает неподкреплённые активные обещания отправки файла/ссылки/фрагмента;
  - делает общий анти-повтор для near-repeat ответов;
  - не переписывает `manager_only`, P0/high-risk, refund/legal/payment-dispute safe templates.
- P0/refund/legal/payment ветки, `codes_from_text`, `pilot_gold_v1`, `no_auto_send` и `manager_approval_required` не менялись.

## Файлы

- `src/mango_mvp/channels/subscription_llm_parts/support.py`
- `src/mango_mvp/channels/subscription_llm_parts/direct_path.py`
- `src/mango_mvp/channels/subscription_llm_parts/policy_routing.py`
- `src/mango_mvp/channels/subscription_llm_parts/post_layers.py`
- `src/mango_mvp/channels/subscription_llm_parts/__init__.py`
- `tests/test_subscription_llm_draft_provider.py`

## Тесты

- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_subscription_llm_draft_provider.py -k "prose_model_led"`  
  Результат: `9 passed, 494 deselected`.
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q`  
  Результат: `3503 passed, 5 skipped, 1 warning in 47.29s`.

Предупреждение: `urllib3 NotOpenSSLWarning` из локального Python/LibreSSL, не связано с изменением.

## Замер OFF -> ON

Набор: `/Users/dmitrijfabarisov/Claude Projects/Foton/2026-06-21_NABOR_chastye_voprosy_min.jsonl`  
SHA256: `7e697be274ebda46087b187f365f14dcbdfadccfa05ab4bb986d4ca56c3ae528`  
Судья: `v9.1`, `parallel 4`, временный `CODEX_HOME` с `service_tier=fast`.  
Профиль `pilot_gold_v1` не включался; direct-path профиль задан явными env-флагами. Снапшот KB: `kb_release_20260612_v6_7_staging_r4_1`.

### OFF

Папка: `runs/20260622_prose_model_led_freq20_OFF`

- `config_validity.invalid=false`
- `dialogs=20`, `turns=70`
- `PASS=13`, `PASS_WITH_NOTES=6`, `FAIL=1`
- `hard_gate_failures=1`
- `bot_answer_self_for_pilot`: `35/70`
- машинные счётчики:
  - `robot_opening=0`
  - `places_template=0`
  - `manager_placeholder=0`
  - `any_brackets=0`
  - `meta_fact=2`
  - `fake_send=0`
  - exact-repeat диалоги: `[]`

FAIL: `sm_u_install`, причина судьи: клиенту показана служебная мета-фраза «в фактах нет».

### ON v2

Папка: `runs/20260622_prose_model_led_freq20_ON_v2`

- `config_validity.invalid=false`
- `dialogs=20`, `turns=73`
- `PASS=8`, `PASS_WITH_NOTES=12`, `FAIL=0`
- `hard_gate_failures=0`
- `bot_answer_self_for_pilot`: `25/73`
- машинные счётчики:
  - `robot_opening=0`
  - `places_template=0`
  - `manager_placeholder=0`
  - `any_brackets=0`
  - `meta_fact=0`
  - `fake_send=0`
  - exact-repeat диалоги: `[]`

Диагностический первый ON-прогон `runs/20260622_prose_model_led_freq20_ON` был отброшен после регрессий: модель написала неподтверждённое «прикрепляю фрагмент», привязала адрес к конкретной группе и вывела мета-фразу «в фактах нет». Эти классы закрыты в ON v2 тестами и гардом.

## ACK

- Live/AMO/Tallanto не трогались.
- `pilot_gold_v1` не менялся.
- P0/refund/legal/payment ветки не редактировались.
- Качество/катимость не оцениваю: вердикт по сырью за Claude #1.
