# PLAN: refactor `subscription_llm.py`

Дата: 2026-06-11
Статус: `phase0_read_only_plan_for_architect_audit`
Рабочее дерево: `/Users/dmitrijfabarisov/Projects/Mango analyse.refactor`
Ветка: `codex/refactor-subscription-llm`
Текущая база после rebase: `61fb34ca Report pilot git cleanup`

## 0. Границы

Это только фаза 0: чтение кода и план. Реализацию переноса не начинать.

Старт переноса допустим только после:

1. git-уборки основного проекта и подтверждения, что нет смешения треков;
2. аудита архитектора и второго архитектора;
3. отдельной отмашки Дмитрия;
4. чистого рабочего дерева в refactor-worktree.

Проверка worktree выполнена:

- стартовая точка фазы 0 была `23336e3b`;
- перед фазой 1 выполнен `git fetch origin main`;
- `origin/main` оказался другой веткой истории для этого worktree и дал конфликтный replay 399 старых коммитов; этот rebase был остановлен;
- целевой rebase выполнен на локальный `main`, как указано в ТЗ: `61fb34ca`;
- в `src/mango_mvp/channels/dialogue_memory.py` есть `TELEGRAM_MEMORY_PROVENANCE`;
- работа велась в `/Users/dmitrijfabarisov/Projects/Mango analyse.refactor`.

## 0.1. Ответы архитектора

Решения, обязательные для фаз 1-5:

1. `src/mango_mvp/channels/subscription_llm.py` остается фасадом на весь период переноса. Он реэкспортирует все имена, включая приватные. Судьба фасада решается отдельно после волны 8.
2. Порядок волн подтвержден: `characterization -> leaves -> DTO -> cleanup -> direct -> policy/routing -> post-layers -> provider`.
3. Отдельный public API module не нужен. Достаточно `contracts.py` из этого плана.
4. Тестовые импорты не переписывать до конца переноса. Менять можно только точки патчей из seam-списка ниже, с фиксацией каждой правки в отчете волны.
5. `constants.py` допустим как механический перенос при правилах раздела 11.1: байты текстов/regex не менять, имена не менять.

## 0.2. Жесткий порядок старта фазы 1

Порядок перед baseline freeze:

1. Rebase на актуальный `main`.
2. Регенерация AST-карты.
3. Diff регенерированной карты против раздела 3 плана.
4. Скрипт автосверки: список top-level имен монолита равен реэкспортам фасада.
5. Только после этого baseline freeze.

Перезаморозка baseline после старта запрещена. Если baseline оказался неверным, волна останавливается и уходит архитектору, а не перезаписывается.

Текущий drift после rebase:

| Метрика | Фаза 0 | После rebase на `main` |
|---|---:|---:|
| Строк | 13 463 | 13 522 |
| AST top-level items с импортами | 793 | 799 |
| Top-level functions/classes | 474 / 8 | 476 / 8 |
| Top-level assignment names | 278 | 282 |
| Uppercase top-level constants | 273 | 277 |
| Локальные `*_ENV` | 49 | 52 |
| Regex `*_RE` | 82 | 82 |
| Text-like constants | 94 | 94 |

Новые/сдвинутые зоны после rebase связаны с memory provenance compact, PII relation stopwords и child ellipsis flags. Раздел 3 остается исторической картой фазы 0; перед baseline freeze обязателен generated AST artifact по текущему файлу.

## 1. Source of Truth

Перед картированием прочитаны:

- `AGENTS.md`;
- `docs/CURRENT_STATE.md`;
- `docs/DECISIONS_LOG.md`;
- `docs/ROADMAP.md`;
- `docs/RUNBOOK.md`;
- `docs/TZ_TELEGRAM_PILOT_JOURNAL_AND_DIALOGUE_STRATEGY_2026-05-23.md`;
- `docs/TZ_DIALOGUE_MEMORY_AND_FAILURE_SKILLS_2026-05-23.md`;
- `docs/SEMANTIC_REVIEW_RULES.md`;
- список последних файлов в `audits/_inbox/`.

Ограничения: не запускать ASR, Resolve+Analyze, live AMO/CRM/Tallanto write, public Telegram polling, тяжелые batch/run-ui/live scripts, не менять `stable_runtime`.

## 2. Факты по файлу

Файл: `src/mango_mvp/channels/subscription_llm.py`

Измерения фазы 0 на базе `23336e3b`:

| Метрика | Значение |
|---|---:|
| Строк | 13 463 |
| Размер | около 636 KB |
| AST top-level items с импортами | 793 |
| Top-level functions/classes | 474 / 8 |
| Top-level assignment names | 278 |
| Uppercase top-level constants | 273 |
| Import statements / imported aliases | 33 / 82 |
| Локальные `*_ENV` | 49 |
| `*_ENV` импортировано из `tone_block` | 4 |
| `os.getenv` calls / `os.environ` checks | 46 / 7-8 |
| Листья без локальных top-level вызовов | 206 |

Актуальные после rebase измерения зафиксированы в разделе 0.2; все дальнейшие baseline/golden строятся только от состояния после rebase на `61fb34ca`.

## 3. Карта файла

| Строки | Размер | Блок | Категории | Назначение |
|---:|---:|---|---|---|
| 1-85 | 85 | imports | pipeline imports | Импорты stdlib, dialogue contract, rules engine, tone, humanity, sanitizers, question catalog. |
| 88-906 | 819 | constants | константы/флаги/gates/render | Env-флаги, схемы, safe-тексты, regex, маршруты, high-risk маркеры, topic sets. |
| 910-1015 | 106 | result DTO | render/утилиты | `SubscriptionDraftResult`: нормализация route/text/flags/metadata и `to_json_dict()`. |
| 1019-1273 | 255 | safe template spec | gates/render/routing | `SafeTemplateSpec`, применение safe-template и выбор route. |
| 1276-2101 | 826 | rules bridge | pipeline/routing/gates | Dialogue-contract mapping, migrated rules engine, selling signals, template dispatcher. |
| 2104-2204 | 101 | direct toggles | prompt/флаги | Direct Path mission/rubric/toggles. |
| 2207-2519 | 313 | direct facts | pipeline/render/утилиты | Snapshot/fact extraction, template-from-KB, trace events. |
| 2522-2886 | 365 | fact pack | render/утилиты | Категории facts, scoring, fact-pack, render fact block. |
| 2889-3107 | 219 | retriever | prompt/pipeline | Prompt для LLM retriever, выбор fact ids, fallback fact-pack. |
| 3110-3288 | 179 | presale prompt safety | prompt/gates | Recent messages, known slots, PII-safe memory view. |
| 3291-3414 | 124 | gold examples | prompt/pipeline | Загрузка YAML и подбор few-shot примеров. |
| 3417-3773 | 357 | direct prompt/finalize | prompt/gates/render | Основной Direct Path prompt, P0 preblock, metadata, finalize. |
| 3776-4790 | 1015 | provider core | routing/pipeline/gates | `SubscriptionLlmDraftProvider`: ветки direct path/dialogue contract/legacy, Codex exec, guard chain, cache. |
| 4793-5012 | 220 | fake/codex/normalize | утилиты/pipeline | Fake provider, Codex command/env/config, payload normalization, fallback draft. |
| 5029-5600 | 572 | A2/tone close | render/gates/routing | A2 proactive, tone sell observer, close-detect, phone/time/rich-format guards. |
| 5600-7023 | 1424 | output gate | gates/render | Authoritative output gate, night-hours note, output sanitizer, PII echo sanitizer, finding helpers. |
| 7026-7841 | 816 | parsing/basic guards | утилиты/gates | JSON parse, internal markers, identity/promo/placeholder guards, recovery candidates. |
| 7844-8795 | 952 | humanity/semantic | gates/prompt/render | Humanity guards, X2 rewriter, phase2 tone, semantic verifier prompts, diagnosis guard. |
| 8798-9568 | 771 | policy/high-risk | gates/routing | Subscription/input policy, high-risk content, autonomy matrix, payment confirmation. |
| 9571-10362 | 792 | intent/context guards | gates/routing/утилиты | Conversation intent, known-context, unstated subject, brand/taxonomy, risk predicates. |
| 10365-11587 | 1223 | safe template library | gates/render | Safe templates, scope fact guard, helpful templates, price/date softening. |
| 11590-11880 | 291 | route decision | routing/утилиты | Autonomy policy, `RouteDecision`, `decide_route`, memory/current-fact helpers. |
| 11883-12727 | 845 | humanity facts/payment | render/gates/утилиты | Humanity factual answers, precise facts, PII/price/payment helpers. |
| 12730-13126 | 397 | shared helpers | pipeline/утилиты | Active brand/topic, dialogue-contract style/safety flags, fact matching anchors. |
| 13129-13356 | 228 | toggles | флаги/gates | Pilot/profile/env toggle helpers, answer-quality, humanity X2 repo gate. |
| 13359-13463 | 105 | aliases/tail | API/утилиты | Backward-compatible aliases, safety contract, clean helpers, cache guard, retry errors. |

Смешанная ответственность:

- `SubscriptionLlmDraftProvider.build_draft` выбирает pipeline, вызывает LLM, применяет safety, tone, semantic verifier и output gate.
- Direct Path блок смешивает prompt, retrieval, KB/snapshot чтение, PII-фильтрацию, few-shot examples, preblock и metadata.
- `apply_high_risk_content_guards` одновременно детектит P0, выбирает safe templates, правит route/text/flags.
- `apply_authoritative_output_gate`, `apply_output_sanitizer`, semantic verifier и high-risk guards пересекаются по изменению текста и маршрута.
- Текстовые константы содержат бизнес-факты: цены, даты, бренды, документы, скидки, P0-safe формулировки.

## 4. Внутренние вызовы

AST-карта:

- top-level functions/classes: 482;
- локальных edges между top-level callable: 914;
- leaf callable без локальных вызовов: 206;
- public names: около 60.

Главные узлы:

| Узел | Роль | Локальные связи |
|---|---|---:|
| `SubscriptionLlmDraftProvider` | главный orchestration | 79 callees |
| `apply_high_risk_content_guards` | самый плотный policy gate | 33 callees |
| `FakeSubscriptionLlmDraftProvider` | fake legacy provider с guard chain | 21 callees |
| `apply_humanity_guards` | post-draft humanity repairs | 20 callees |
| `_authoritative_gate_findings` | сбор finding для final gate | 17 callees |
| `_apply_migrated_rules_engine` | bridge rules engine | 15 callees |
| `apply_dialogue_contract_v2_template_dispatcher` | dispatcher safe templates | 15 callees |
| `apply_autonomy_matrix_guard` | route autonomy guard | 13 callees |

Самые общие helpers по входящим вызовам:

- `_active_brand`: 37 callers;
- `_truthy_value`: 33 callers;
- `_conversation_intent_plan`: 17 callers;
- `_humanity_previous_bot_texts`: 12 callers;
- `is_high_risk_result`: 11 callers;
- `known_context_fields`: 10 callers.

Ключевые пути:

```text
SubscriptionLlmDraftProvider.build_draft
  -> direct path:
     _build_direct_path_draft
       -> _direct_path_context_fact_pack
       -> _build_direct_path_prompt
       -> _direct_path_draft_runner
       -> apply_semantic_output_verifier
       -> apply_authoritative_output_gate

  -> dialogue contract:
     _build_dialogue_contract_pipeline_draft
       -> _apply_dialogue_contract_v2_guard_chain
       -> apply_dialogue_contract_v2_template_dispatcher
       -> _dialogue_contract_v2_route_permission_guard
       -> decide_route

  -> legacy:
     build_draft_prompt
       -> generate_from_prompt
       -> _run_once
       -> normalize_subscription_draft_payload
       -> guard chain, rewriter, autonomy, humanity, verifier, final gate
```

Циклы:

- `_presale_prompt_safe_slot_value -> _presale_prompt_safe_value -> _presale_prompt_safe_mapping -> _presale_prompt_safe_slot_value`;
- `_presale_prompt_safe_value` рекурсивна сама в себя;
- `_mapping_has_client_safe_current_fact` рекурсивна;
- `_append_fact_texts` рекурсивна.

## 5. Внешние вызовы из `src/` и `scripts/`

Прямой grep/AST по `src/` и `scripts/`, без самого `subscription_llm.py`, нашел 34 импортируемых символа.

Критичный runtime/API:

| Символ | Кто зовет | Риск переноса |
|---|---|---|
| `SubscriptionDraftResult` | `src/mango_mvp/pilot_context_assembly.py`, `src/mango_mvp/integrations/draft_loop.py`, `src/mango_mvp/channels/answer_quality_rewriter.py` только typing, Telegram/eval scripts | Высокий: главный DTO, нужен compatibility shim. |
| `SubscriptionLlmDraftProvider` | `scripts/run_telegram_public_pilot_bots.py`, `scripts/run_amo_wappi_draft_loop.py`, `scripts/telegram_manager_draft_pilot.py`, `scripts/run_telegram_night_shadow_replay.py`, `scripts/run_telegram_dynamic_client_sim.py`, `scripts/run_telegram_stage6_kb_eval.py`, `scripts/run_telegram_pilot_concurrency_smoke.py` | Высокий: операционные и replay скрипты. |
| `AUTONOMY_MATRIX_SAFE_TOPIC_IDS` | `src/mango_mvp/pilot_context_assembly.py`, `scripts/run_telegram_dynamic_client_sim.py`, `scripts/run_telegram_night_shadow_replay.py` | Средний: матрица автономности и контекст. |
| `SAFE_FALLBACK_DRAFT_TEXT` | `scripts/run_telegram_public_pilot_bots.py` | Средний: публичный fallback. |
| `DIRECT_PATH_PILOT_CONFIG_ENV`, `DIRECT_PATH_PILOT_CONFIG_VERSION`, `DIRECT_PATH_REAL_MANAGER_GOLD_PACK_VERSION` | `scripts/run_amo_wappi_draft_loop.py` | Средний: direct-path pilot config. |

Eval/test API в `scripts/`:

| Символы | Кто зовет | Риск |
|---|---|---|
| `normalize_subscription_draft_payload` | dynamic sim, stage6 eval, concurrency smoke | Средний для replay/smoke. |
| `strip_internal_service_markers` | public bots, dynamic sim, concurrency smoke | Средний: влияет на публичный текст. |
| `build_codex_exec_command`, `build_codex_exec_env`, `codex_isolation_cwd` | question catalog codex full run, dynamic sim | Низкий-средний: tooling. |
| `apply_autonomy_matrix_guard`, `apply_brand_separation_guard`, `apply_high_risk_content_guards`, `apply_input_policy_guards`, `apply_payment_confirmation_guard`, `apply_unsupported_promise_guard` | `scripts/run_telegram_stage6_kb_eval.py` | Средний для воспроизводимости stage6 eval. |

Только re-export в `src/mango_mvp/channels/__init__.py`, без реальных вызовов в `src/`/`scripts/`:

`CodexExecConfig`, `CodexExecDraftProvider`, `DEFAULT_CODEX_MODEL`, `DEFAULT_CODEX_REASONING_EFFORT`, `DraftGenerationResult`, `FakeDraftProvider`, `FakeSubscriptionLlmDraftProvider`, `SUBSCRIPTION_LLM_SCHEMA_VERSION`, `contains_bot_identity_disclosure`, `draft_has_identity_disclosure`, `extract_json_object`, `find_identity_disclosure_phrases`, `guard_identity_disclosure`, `parse_llm_json`, `safe_fallback_draft`, `subscription_llm_safety_contract`.

Важно: тесты импортируют много приватных `_...` имен. Пока тесты не переписаны, старый `subscription_llm.py` должен оставаться фасадом и реэкспортировать даже приватные имена, которые используются в тестах.

## 6. Предложение разбиения

Рекомендуемый формат: пакет

```text
src/mango_mvp/channels/subscription_llm_parts/
```

Старый файл `src/mango_mvp/channels/subscription_llm.py` на период миграции остается compatibility facade. Внешние импорты из `mango_mvp.channels.subscription_llm` должны продолжать работать.

Целевая структура:

| Файл | Ответственность |
|---|---|
| `constants.py` | Схемы, env names, route/topic sets, safe text constants, regex constants. |
| `contracts.py` | `SubscriptionDraftResult`, `SafeTemplateSpec`, `RouteDecision`, public aliases. |
| `utils.py` | `_clean_list`, `_clamp_float`, `_optional_text`, `_truthy_value`, `_active_brand`, shared text/list helpers. |
| `codex_exec.py` | `CodexExecConfig`, `build_codex_exec_command`, `codex_isolation_cwd`, `build_codex_exec_env`, retry/cache guard helpers. |
| `parsing.py` | `extract_json_object`, `parse_llm_json`, `normalize_subscription_draft_payload`, `safe_fallback_draft`. |
| `output_cleanup.py` | internal marker stripping, output sanitizer, PII echo sanitizer. |
| `fact_support.py` | fact text extraction, anchor matching, `_claim_supported_by_facts`, `_keep_answer_supported`. |
| `direct_facts.py` | direct path snapshot/fact pack/retriever ids/category scoring. |
| `direct_prompt.py` | direct path prompt, LLM retriever prompt, presale prompt safety, gold examples. |
| `direct_metadata.py` | direct path metadata, preblock, prepare/finalize. |
| `safe_templates.py` | safe template registry, terminal/cross-brand/refund/scope/missing-fact templates. |
| `rules_bridge.py` | migrated rules engine adapter and selling signals. |
| `policy_guards.py` | payment/input/high-risk/autonomy/funnel/brand/taxonomy/known-context guards. |
| `route_decision.py` | autonomy policy, `decide_route`, memory follow-up helpers. |
| `semantic_verifier.py` | semantic output verifier/regen prompts, diagnosis prompt/guard. |
| `humanity_layers.py` | humanity guards, X2 rewriter, precise fact answers, humanity helpers. |
| `tone_layers.py` | A2 proactive, tone sell observer, tone close, night hours note. |
| `authoritative_gate.py` | final authoritative output gate and findings. |
| `provider.py` | `SubscriptionLlmDraftProvider`, `FakeSubscriptionLlmDraftProvider`, provider runner methods, cache get/put. |

## 7. Волны переноса

Фаза 1 после отмашки не должна переносить рискованные поверхности. Сначала нужен characterization harness и golden replay.

| Волна | Что делать | Что не трогать | Byte-for-byte критерий | Проверка |
|---|---|---|---|---|
| 1. Characterization only | Добавить baseline/golden harness, зафиксировать список публичных и тестовых импортов, собрать фиксированные replay cases. | Константы, env-флаги, pipeline imports, тексты, regex, порядок guard chain. | Текущий monolith дает стабильные prompt bytes и canonical result JSON на фиксированных входах. | Full pytest + replay baseline. |
| 2. Parser/codex pure leaves | Вынести `codex_exec.py`, часть `parsing.py`, retry/cache guard helpers, без изменения public import path. | `SubscriptionDraftResult`, safe texts, env fallback helpers. | `build_codex_exec_command/env`, `extract_json_object`, `parse_llm_json`, fallback JSON идентичны. | Full pytest + replay legacy prompt. |
| 3. DTO/contracts | Вынести `SubscriptionDraftResult`, `CodexExecConfig`, aliases, минимальные route constants как re-export. | Не менять `__post_init__`, key order `to_json_dict()`, порядок flags. | `to_json_dict()` и normalized payload byte-for-byte одинаковы. | Full pytest + DTO golden fixtures. |
| 4. Output cleanup/fact support | Вынести marker stripping, identity guards, PII sanitizer, fact anchor helpers. | Не редактировать regex и тексты; перенос только механический. | Cleaned text, flags, metadata and fact-support decisions identical. | Full pytest + focused guard replay. |
| 5. Direct path leaves | Вынести direct fact-pack, retriever prompt, gold examples, direct prompt, direct metadata. | Не менять `DIRECT_PATH_*`, `lru_cache`, `template_from_kb_trace`, snapshot path handling. | Fact pack JSON, prompt bytes, metadata JSON identical. | Full pytest + direct-path replay with fake runner. |
| 6. Policy/templates/routing | Вынести safe templates, rules bridge, high-risk/autonomy/payment/brand/funnel guards, `decide_route`. | Не менять порядок guard calls и `dict.fromkeys` ordering. | Route, draft text, safety_flags, checklist, metadata identical for guard replay. | Full pytest + stage6 fake eval replay. |
| 7. Semantic/humanity/tone/final gate | Вынести semantic verifier, diagnosis, humanity, tone, A2, authoritative gate. | Не менять prompt text, verifier action matrix, final downgrade rules. | Verifier prompts, fake verifier results and final gated JSON identical. | Full pytest + verifier/humanity/tone replay. |
| 8. Provider core | Вынести `SubscriptionLlmDraftProvider`, fake provider, runner methods, cache get/put. Старый файл становится thin facade. | Не менять ветвление `direct path -> dialogue contract -> legacy`, не менять runner/caching semantics. | Provider replay по legacy/direct/dialogue/fake дает identical prompt bytes, command args, route, text, flags, metadata. | Full pytest + fixed replay before/after. |

Уточнение состава волн 2-5 после dependency-check:

- Волна 2: только настоящие листья `build_codex_exec_command`, `build_codex_exec_env`, `codex_isolation_cwd`, `_with_codex_exec_metadata`, `_cache_key`, `_guard_cache_dir`, `_is_retryable`, `_CodexRetryableError`, `_PromptProviderError`, `extract_json_object` и их константы `DEFAULT_CODEX_MODEL`, `DEFAULT_CODEX_REASONING_EFFORT`, `_RETRYABLE_MARKERS`. Не переносить `parse_llm_json`, `normalize_subscription_draft_payload`, `safe_fallback_draft`, `_normalize_direct_path_payload`.
- Волна 3: `SubscriptionDraftResult` переезжает вместе с `SUBSCRIPTION_LLM_SCHEMA_VERSION`, `SAFE_FALLBACK_DRAFT_TEXT`, `BASE_SAFETY_FLAGS`, `ALLOWED_ROUTES`, `ALLOWED_MESSAGE_TYPES`, `_clean_list`, `_clean_crm_recommendations`, `_clamp_float` и минимальным strip-блоком `strip_internal_service_markers`/`INTERNAL_*`/`_normalize_output_sanitizer_text`, потому что `__post_init__` зависит от него.
- Волна 4: PII/toggle/fact helpers переносить кластерами. `_truthy_value`, `_explicit_truthy_setting`, `_pilot_profile_*`, `_direct_path_pilot_config`, `_A2_PHONE_RE`, `_CLIENT_EMAIL_RE`, `_looks_like_russian_surname`, `_presale_prompt_child_name_value` должны уйти с первым потребителем, чтобы не создать циклы.
- Волна 5: переносить snapshot/fact-pack/render/retriever/gold/prompt leaves. Не считать leaf: `_direct_path_preblocked_result`, `_direct_path_p0_text`, `_direct_path_finalize_metadata`, `_direct_path_prepare_model_result`, `_normalize_direct_path_payload`; их держать до policy/post-layer/provider волн или переносить только с нужными shared-зависимостями.

Коммитная дисциплина: одна волна = отдельный маленький коммит и audit pack. При любом расхождении replay или pytest волну не продолжать.

Скорректированное правило циклов:

- parts-модули никогда не импортируют фасад `subscription_llm.py`;
- helper переезжает в ту же волну, что его первый parts-потребитель;
- после каждой волны фасад содержит только imports/reexports/`__all__`, без `def`, `class` и новых runtime-веток;
- дублировать определения запрещено: только move + reexport.

Разрешенные seam-точки, которые остаются методами и могут патчиться через `self.*`:

- `runner`;
- `sleep`;
- `_direct_path_draft_runner`;
- `_direct_path_llm_retrieve_runner`;
- `_humanity_x2_rewrite_runner`;
- `_build_dialogue_contract_pipeline_draft`;
- `_cache_get`;
- `_cache_put`.

Любая адаптация тестовой patch-точки вне этого списка запрещена без отдельного решения архитектора.

## 8. Универсальный критерий поведения

Для каждой волны критерий не “тесты зеленые”, а:

```text
input -> prompt bytes -> route + draft_text + safety_flags + metadata
```

должны совпадать byte-for-byte или через заранее описанный canonical JSON, где порядок ключей фиксирован.

Сравнивать:

- `client_message`;
- canonical `context` JSON;
- env snapshot;
- выбранную ветку provider: `legacy`, `direct_path`, `dialogue_contract`, `fake`;
- command args для Codex runner;
- prompt bytes `sha256`;
- raw runner payload;
- canonical `SubscriptionDraftResult.to_json_dict()`;
- `route`;
- `draft_text`;
- `message_type`;
- `topic_id`;
- `safety_flags` в порядке;
- `manager_checklist` в порядке;
- `missing_facts`;
- `forbidden_promises_detected`;
- `metadata` с сохранением key order там, где оно влияет на JSON/golden.

Replay должен быть без live Codex/Telegram/AMO/Tallanto:

- `cache_dir=None`;
- mocked runner пишет заранее заданный JSON в `--output-last-message`;
- фиксированные env;
- фиксированные context/time;
- никакой записи в `stable_runtime`;
- никаких public Telegram `poll`.

## 9. Скелет теста эквивалентности

Минимальная форма будущего теста:

```python
def test_subscription_llm_refactor_equivalence_prompt_route_text(monkeypatch):
    import hashlib
    import json
    import subprocess
    from pathlib import Path

    import mango_mvp.channels.subscription_llm as llm

    captured = []

    def runner(cmd, **kwargs):
        prompt_bytes = kwargs["input"].encode("utf-8")
        captured.append(
            {
                "cmd": list(cmd),
                "prompt_sha256": hashlib.sha256(prompt_bytes).hexdigest(),
                "prompt_bytes": prompt_bytes,
            }
        )
        out = Path(cmd[cmd.index("--output-last-message") + 1])
        out.write_text(
            json.dumps(
                {
                    "message_type": "question",
                    "topic_id": "theme:001_pricing",
                    "route": "draft_for_manager",
                    "draft_text": "Здравствуйте! Сориентирую по стоимости.",
                    "confidence_theme": 0.9,
                    "confidence_group": 0.9,
                    "risk_level": "low",
                    "safety_flags": ["manager_approval_required", "no_auto_send"],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(cmd, 0, "", "")

    context = {
        "active_brand": "foton",
        "TELEGRAM_DIRECT_PATH": "0",
        "TELEGRAM_DIALOGUE_CONTRACT_PIPELINE": "0",
        "answer_quality_llm_rewrite_enabled": False,
        "semantic_output_verifier_enabled": False,
        "rop_policy": {"bot_permission": "draft_for_manager"},
        "known_slots": {"grade": "9", "subject": "физика"},
    }

    provider = llm.SubscriptionLlmDraftProvider(runner=runner, cache_dir=None)
    result = provider.build_draft("Сколько стоит физика 9 класс?", context=context)

    actual = {
        "prompt_sha256": captured[0]["prompt_sha256"],
        "route": result.route,
        "draft_text": result.draft_text,
        "result_json": result.to_json_dict(),
    }
    assert actual == load_expected_case("legacy_pricing_foton_grade9")
```

Для сравнения до/после волны baseline лучше хранить как JSONL:

```json
{
  "case_id": "legacy_pricing_foton_grade9",
  "branch": "legacy",
  "client_message": "Сколько стоит физика 9 класс?",
  "context_sha256": "...",
  "env_sha256": "...",
  "prompt_sha256": "...",
  "prompt_utf8": "...",
  "runner_payload_sha256": "...",
  "route": "draft_for_manager",
  "draft_text": "...",
  "result_json": {...}
}
```

Набор cases:

- legacy prompt, direct path off, dialogue contract off;
- direct path on, LLM retrieve off;
- direct path on, LLM retrieve on with fake retriever;
- dialogue contract pipeline on with fake runners;
- high-risk/P0 input;
- brand separation;
- payment confirmation conflict;
- known-context no re-ask;
- fake provider path.

Обязательные дополнительные cases:

- `DIRECT_PATH_REAL_MANAGER_GOLD_PACK_PATH` без env override: `assert path.exists()` на default path;
- direct path replay без override пути gold pack;
- `valid_until` только пустой или `2099`;
- `now_msk_hour` всегда задан в `context`;
- взрывающийся sentinel на `datetime.now` и `date.today` в фасаде и во всех parts-модулях;
- runner `rc != 0` retryable;
- runner `rc != 0` non-retryable;
- runner timeout;
- cache `put -> hit` в tmp-каталоге;
- `_guard_cache_dir` на пути внутри `stable_runtime`;
- память-слоты с provenance;
- route rubric;
- night-hours note;
- fail on unexpected fallback или неожиданный `metadata.reason`/`reason_class`.

Replay обязан считать тихий fallback ошибкой переноса. Широкие `except Exception` в монолите не должны маскировать import/name/runtime ошибки после разбиения.

## 10. Проверки после каждой волны

Базовый safe collect:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest --collect-only -q
```

Полная локальная проверка после каждой волны:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests
```

Минимальный focused набор перед полным pytest:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q \
  tests/test_subscription_llm_draft_provider.py \
  tests/test_draft_prompt_builder.py \
  tests/test_bot_policy_v2.py \
  tests/test_dialogue_contract_pipeline.py \
  tests/test_telegram_dynamic_client_sim.py \
  tests/test_telegram_public_pilot_bots.py \
  tests/test_telegram_bot_polling.py \
  tests/test_telegram_manager_draft_pilot_script.py \
  tests/test_draft_loop.py
```

Replay:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/run_subscription_llm_equivalence_replay.py \
  --cases D1_audit_backlog/subscription_llm_refactor_cases_2026-06-11.jsonl \
  --baseline D1_audit_backlog/subscription_llm_refactor_baseline_2026-06-11.jsonl \
  --out audits/_inbox/subscription_llm_refactor_replay_<timestamp>
```

Скрипт выше пока не существует; его нужно делать только после аудита плана и отмашки.

Дополнительные обязательные проверки после каждой волны:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/check_subscription_llm_facade_exports.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/check_subscription_llm_move_only.py
```

Скрипт должен проверять:

- top-level имена в монолите до волны равны реэкспортам фасада после волны;
- `facade.X is parts.X` для всех реэкспортов;
- классы, функции и исключения не продублированы;
- особенно `_CodexRetryableError`, `_PromptProviderError`, `SubscriptionDraftResult`, `SubscriptionLlmDraftProvider`.

Move-only checker должен проверять:

- перед baseline freeze заморозить normalized AST и compiled body для каждой top-level функции/класса монолита;
- после появления `subscription_llm_parts` сравнивать каждую функцию/класс из frozen snapshot с телом в parts;
- маскировать имя файла, номера строк и позиции docstring;
- любое отличие тела = fail волны даже при зеленых pytest/replay;
- допустимые отличия только в import-секциях, их diff печатается в отчет волны;
- дубли definition name в parts = fail.

Coverage parity:

- focused pytest и replay гоняются под coverage по пакету `subscription_llm_parts`;
- правило: строки, покрытые до волны, покрыты после волны;
- падение coverage parity равно падению волны.

Протокол волны:

1. Перенос.
2. Focused pytest.
3. Full pytest.
4. Replay vs frozen baseline.
5. Coverage parity.
6. Identity asserts.
7. Отчет-страница в `audits/_inbox/<wave>_<timestamp>/`.
8. Коммит.

Отчет волны должен содержать:

- что переехало;
- diff-статистику;
- список адаптированных seam/patch-точек;
- focused/full pytest output;
- replay output;
- coverage parity output;
- identity asserts output;
- остаточные риски.

Если не совпало что угодно: `git revert` коммита волны, отчет о расхождении и стоп до архитектора. Чинить поверх запрещено.

Постоянное дополнение K: самопочинка после падения wave gate разрешена только для класса A и только test-only:

- класс A: флейк доказан на чистом монолите до переноса;
- допустимая правка: только тестовая обвязка, изоляция env/cache/time/path; прод-код не трогать, assert'ы не ослаблять;
- каждый test-only flake fix фиксировать отдельным коммитом и отдельной строкой в отчете волны;
- после test-only flake fix обязательны два полных зеленых pytest подряд на монолите;
- класс B: монолит зеленый, а после переноса падает full pytest или другой gate; любые правки прод-кода запрещены, нужно откатить волну и остановиться с отчетом архитектору;
- неопределимый класс: не воспроизводится ни на монолите, ни на перенесенной волне за два полных прогона; считать классом B;
- если для классификации нужна диагностика класса B, собрать порядок тестов до падения через `junit-xml` и указать кандидатов, кто трогает env/lru_cache/Path/time без teardown.

Wave2 incident, решение архитектора от 2026-06-11:

- оригинальный diff падения был в `dialogue_memory_state`;
- wave2 переносит только `codex_exec` / parsing / cache leaf-блоки и сборку памяти не трогает;
- чистый монолит после инцидента прошел 10 изолированных прогонов `test_public_pilot_context_matches_extracted_assembly` и два полных pytest подряд;
- reapplied wave2 прошла два полных pytest подряд;
- move-only, identity-asserts и replay на reapplied wave2 зеленые;
- вывод архитектора: это пред-существующий порядко-зависимый флейк тестовой инфраструктуры, вероятно загрязнение env/кэша соседним тестом в worker, а не сигнал эквивалентности wave2;
- решение: продолжать после отдельного test-only fix, который изолирует `TELEGRAM_MEMORY_PROVENANCE`, `TELEGRAM_DIRECT_PATH_PILOT_CONFIG` в проблемном тесте и гарантирует `get_settings.cache_clear()` в `finally` в `tests/test_smoke.py`;
- правило K остается в силе: если порядко-зависимый флейк повторится на любом тесте, собрать junit обоих прогонов и факт, трогает ли волна область diff'а, затем STOP к Дмитрию.

Автономный объем текущего запуска:

1. дополнения в план;
2. волна 1: harness + cases + baseline;
3. волны 2-5;
4. после волны 5 стоп-чекпоинт и сводный отчет архитектору.

Волны 6-8 не начинать без новой отмашки.

Не запускать без отдельного подтверждения:

- `scripts/run_telegram_public_pilot_bots.py --mode poll`;
- full dynamic sim в реальном Codex/Claude mode;
- AMO/Wappi live write;
- night shadow replay CLI без fake provider;
- любые batch/live scripts.

## 13. Финальная эквивалентность после волны 8

Перед merge и после завершения волны 8 нужен отдельный финальный replay:

- старое дерево: точка до волны 1, frozen baseline commit;
- новое дерево: итоговый refactor commit;
- полный набор 89 сценариев;
- fake-runner с записанными ответами модели, без live LLM/ASR/AMO/Tallanto/write;
- сравнение по каждому ходу: sha256 prompt bytes, route, draft_text, safety_flags, manager_checklist;
- отчет-таблица расхождений в audit pack финальной приемки;
- ожидаемое число строк расхождений: 0.

Этот отчет является входным артефактом финальной приемки архитектора. Если есть хотя бы одно отличие, merge запрещен до отдельного решения архитектора.

## 11. Риски: не трогать в фазе 1

### 11.1. Теневая база констант

Инвентаризация:

- 94 top-level текстовых шаблона по суффиксам `_TEXT/_TEXTS/_VARIANTS`;
- 78 узких клиентских safe/helpful/handoff шаблонов в основном блоке строк 194-485;
- 102 русскоязычных top-level константы, если включать mission/rubric/category aliases и tone-close тексты;
- 82 top-level regex `*_RE` по локальному счетчику.

Риск: эти константы являются не просто техническими строками. В них зашиты цены, даты, документы, бренды, P0-safe формулировки и политика ответа.

План для фазы 1:

- не переносить клиентские тексты в YAML/KB;
- не дедуплицировать похожие safe-тексты;
- не менять пробелы, `ё`, тире, пунктуацию;
- не переименовывать константы;
- не менять regex/pattern constants;
- сначала зафиксировать golden `prompt_sha256` и `draft_text`.

### 11.2. Флаги и переключатели

Заявленная зона риска: около 111 флагов. Фактическая поверхность распадается на несколько классов:

- 49 локальных `*_ENV`;
- 4 env-константы импортированы из `tone_block`;
- 46 `os.getenv` calls;
- 7-8 `os.environ` checks;
- 59 строгих safety flag literals по счетчику `*FLAGS + safety_flags=`;
- значительно больше metadata/context code keys, если считать все строковые флаги и диагностические ключи.

Риск: env/default логика не плоская. Часто порядок такой:

```text
explicit context override -> os.environ -> pilot profile default -> hardcoded fallback
```

План для фазы 1:

- не переименовывать env vars;
- не менять context aliases;
- не менять metadata keys;
- не менять `DIRECT_PATH_PILOT_PROFILE_DEFAULT_ON_FLAGS`;
- не менять порядок fallback-логики;
- не менять порядок `safety_flags`, особенно сборку через `dict.fromkeys(...)`;
- сделать отдельный generated registry флагов только как read-only audit artifact, без использования runtime.

### 11.3. Pipeline imports

Рискованные импорты:

- `dialogue_contract_pipeline`: 19 aliases, включая приватные `_GENERIC_HANDOFF_TEXTS`, `_handoff_factual_claim_text`, `_HANDOFF_EXHAUSTED_TEXTS`, `_is_pure_handoff_text`, `_established_topic_from_context`;
- `rules_engine`: `RuleOutcome`, `apply_rule`, `load_rules_registry`, `select_rule`;
- `answer_quality_rewriter`, `humanity_*`, `tone_block`, `p0_recall_spec`, `fact_scope_spec`, `semantic_roles`;
- `question_catalog.classifier.load_valid_theme_and_service_ids`.

Связь с KB не прямым импортом, а через `context`: `confirmed_facts`, `facts_context`, `knowledge_snippets`, `gold_answer_context`, `snapshot_path/knowledge_snapshot_path/kb_snapshot_path`.

План для фазы 1:

- не менять imports из `dialogue_contract_pipeline`;
- не заменять приватные imports на новые public API без отдельного ТЗ;
- не делать lazy imports;
- не менять `lru_cache` и snapshot/fact pack поведение;
- не менять `template_from_kb_trace`, потому что это observable metadata;
- не менять `_guard_cache_dir` и запрет cache внутри `stable_runtime`.

## 12. Рекомендуемый результат аудита

Архитектору нужно подтвердить или поправить:

1. Достаточно ли оставить `subscription_llm.py` фасадом на весь период переноса.
2. Согласен ли порядок: characterization first, затем leaves, direct path, guards, post-layers, provider.
3. Нужен ли отдельный public API module для `SubscriptionDraftResult` до переноса provider.
4. Какие приватные тестовые импорты переписать до волны 4, чтобы не тащить весь `_...` API через фасад.
5. Считать ли `constants.py` допустимым механическим переносом или держать константы в старом файле до конца.

До ответов на эти вопросы реализацию не начинать.
