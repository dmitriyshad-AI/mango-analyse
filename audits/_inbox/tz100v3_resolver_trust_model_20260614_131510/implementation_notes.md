# TZ100 v3 resolver trust model on names

Дата: 2026-06-14.

Сделано:

- `PROMPT_VERSION` поднят до `v3`; старый LLM cache не переиспользуется для нового промпта.
- В prompt/schema/normalize/validate/trace проведено поле `merge_confidence: high|low`.
- `additionalProperties: false` сохранён на root и child JSON schema; normalize теперь также reject-ит лишние/отсутствующие ключи для OpenAI-compatible пути.
- `merge_confidence` только логируется и агрегируется; name-вито `different_child_names_merged` и `mention_name_misattributed` не ослаблены.
- Для name-вито добавлен локальный raw diagnostic JSONL через `PROFILE_LLM_CHILD_RESOLVER_NAME_DIAGNOSTICS_PATH`; файл не копируется в git/audit pack, потому что содержит реальные написания имён.
- `name_variants` дедупятся по нормализованному написанию.
- Скрипт микропробы обновлён под v3: default 150, новый `tz100_microprobe_v3_*` out-root, clean-cache guard, known-bad focus, anonymized manual-review files.

Микропроба:

- out-root: `/Users/dmitrijfabarisov/Projects/Mango_tz24_dedup/product_data/customer_profiles/tz100_microprobe_v3_20260614_124341`
- selected_count: 120
- trace_events: 120
- name-veto raw diagnostics: 21 rows, local-only path `name_veto_diagnostics.local.jsonl`
- known-bad found: 4/4
- resolver summary: 86 resolved, 33 fail-soft, 1 shared-phone skipped, 0 cache hits, prompt_version `v3`
- confidence children: high=92, low=105, missing=1 in trace summary; resolver child summary high=92, low=105, missing=0

Go/no-go по 4 known-bad:

- `588aa705`: rejected, confidences `high, high, low`
- `d5ab113b`: accepted, confidences `low, high`
- `daf16c4b`: rejected, confidences `high, high`
- `e55507f6`: rejected, confidences `high`

Вывод: `merge_confidence` пока нельзя использовать как основание для ослабления name-вито. На known-bad есть `high`, `known_bad_all_have_low=false`, `known_bad_any_high=true`.
