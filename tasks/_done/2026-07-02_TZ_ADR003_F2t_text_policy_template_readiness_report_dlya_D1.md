Ветка: codex/adr003-semanticframe-migration
Зоны: scripts/report_adr003_frame_calibration_queue.py, tests/test_report_adr003_frame_calibration_queue.py, tasks/_done/2026-07-02_TZ_ADR003_F2t_text_policy_template_readiness_report_dlya_D1.md
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_report_adr003_frame_calibration_queue.py tests/test_kb_distribution_packs.py tests/test_adr003_regex_understanding_moratorium.py
Семантический-аудит: да

# ADR-003 F2t: text-policy/template readiness report

## Контекст

F2s показал, что часть `proof_reconciliation` строк имеет свежий `client_safe_text`,
но контракт bot-pack запрещает дословно подставлять `client_safe_text` клиенту.
Для активного ответа нужен отдельный слой текстовой политики/шаблонов.

## Решение

Расширить только отчёт `report_adr003_frame_calibration_queue.py`:

- принимать `--bot-template-registry` опционально;
- если аргумент не передан, искать sibling `*_bot_pack/bot_template_registry.json` рядом с `kb_snapshot`;
- индексировать шаблоны по `fact_key` и `fact_id`;
- для каждой proof-reconciliation строки считать:
  - `direct_quote_forbidden`;
  - `template_registry_status`;
  - `text_policy_readiness_status`;
  - `structured_value_available`;
  - длину/хэш шаблона без экспорта `template_text`;
  - ключи `structured_value`, но не значения.

## Границы

- Не менять runtime/provider/direct_path/live/profile.
- Не генерировать клиентский ответ.
- Не экспортировать `client_safe_text`, `fact_text`, `manager_display_text`, `manager_check_text`, `template_text`, `structured_value.raw_value`.
- Не добавлять regex-понимание смысла.
- `active_behavior_allowed=false`, `active_readiness=no_go`.

## Локальный замер

На F2q retry:

- `proof_reconciliation_would_reconcile=17`;
- `direct_quote_forbidden=17`;
- `structured_value_available=17`;
- `source_text_ready_requires_nonquote_policy=8`;
- `template_registry_found_requires_renderer=3`;
- `blocked_wrong_brand=6`;
- `template_registry_found=3`;
- `active_readiness=no_go`.

На свежем M1 `36ea110` proof-reconciliation trace отсутствует, поэтому readiness остаётся `0`.

## Вывод

До active-route/self-answer нужен следующий отдельный слой: shadow renderer/text policy,
который соберёт человеческую фразу из шаблона/structured_value/контекста, а затем пройдёт semantic verifier.
