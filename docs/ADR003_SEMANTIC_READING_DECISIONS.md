# ADR-003 Semantic Reading Decisions

Рабочий журнал решений по ТЗ `2026-07-03_TZ_USKORENIE_semantic_reading_odin_blok_dlya_D1.md`.

Правило: каждое существенное решение фиксируется с обоснованием, сырьём и статусом аудита.

## 2026-07-03

### D1. Не перемещать оригинальный Foton-ТЗ

- Решение: оставить `/Users/dmitrijfabarisov/Claude Projects/Foton/2026-07-03_TZ_USKORENIE_semantic_reading_odin_blok_dlya_D1.md` на месте; в репозитории создать рабочую обёртку в `tasks/_inbox_codex/`.
- Обоснование: `scripts/task_move.py --take` для абсолютного файла перемещает исходник и удаляет его из Foton. Это сломало бы общий источник для Дмитрия и Claude #1.
- Сырьё: `scripts/task_move.py` читает `src`, пишет копию в `tasks/_running`, затем делает `src.unlink()`.
- Аудит: PASS, подтверждено субагентом-аудитором по `scripts/task_move.py`.

### D2. Граница исполнения текущего захода

- Решение: исполнять Э0/Э1 и подготовку Э2; не выполнять Э3-A/Э3-B без регрейда Claude #1 и отдельных решений Дмитрия.
- Обоснование: Э0/Э1/Э2 являются default-OFF/read-only/shadow-фундаментом; Э3-A меняет активные маски классов, Э3-B удаляет legacy-regex и требует отдельного согласия.
- Сырьё: исходное ТЗ, разделы `Э3`, `Роли и порядок`, `Правки по ревью D1`.
- Аудит: PASS, подтверждено субагентом-аудитором по рабочей обёртке ТЗ.

### D3. Э0 удаляет только мёртвые post_layers regex и усиливает страж

- Решение: удалить из `post_layers.py` 9 regex-объявлений (`HIGH_RISK_INPUT_PATTERNS` = 4 паттерна, `LEGAL_CONTEXT_INPUT_RE`, `ZERO_COLLECT_DRAFT_RE`, `REFUND_FORBIDDEN_DETAIL_RE`, `COMPLAINT_APOLOGY_RE`, `COMPLAINT_DETAIL_COLLECT_RE`) и их пассивные импорты/exports; параллельно расширить ADR-003 moratorium guard на marker-helper calls и marker/keyword table snapshots.
- Обоснование: эти regex сейчас не исполняются на direct path; оставлять их в бюджете вредно, потому что они маскируют новый regex-долг. Усиление guard закрывает дыру, где `has_marker`-таблицы могли добавляться вне снапшота.
- Сырьё: `rg` по `src/mango_mvp/channels/subscription_llm_parts` показывает только объявления в `post_layers.py`, импорты в `provider.py`/`monolith.py`, export в `__init__.py` и fixture-снапшоты; рабочих `.search`/вызовов этих имён нет.
- Аудит: PASS_WITH_NOTES; аудитор подтвердил комплект удаления, но потребовал post-implementation `rg` по всему `src/` и ratchet вместо простого grep для `active_behavior_allowed`.

### D4. Расширить pattern snapshot до тяжёлых marker-helper потребителей

- Решение: добавить в `DIRECT_PATH_PATTERN_FILES` не только `text_signals.py` и `policy_routing.py`, но и тяжёлых потребителей marker-helper логики: `answer_quality_rewriter.py`, `new_lead_funnel.py`, `actions.py`, `held_state.py`, `dialogue_memory.py`. Отдельный marker-helper budget теперь считается по всем `src/mango_mvp/channels/**/*.py`, а не только по direct-path snapshot-файлам.
- Обоснование: обновлённое Foton-ТЗ прямо указывает, что без этих файлов marker-понимание остаётся невидимым. Если оставить только `policy_routing.py`, новый regex-долг снова будет расползаться через соседние модули.
- Сырьё: `rg has_any_marker|has_marker` по перечисленным файлам показывает активные потребители; текущий budget зафиксировал 327 marker-helper вызовов по runtime channels.
- Аудит: PASS_WITH_NOTES; первоначальный аудит подтвердил `policy_routing.py`, а после новых материалов Клода решение расширено до полного безопасного Э0-guard.

### D5. `active_behavior_allowed=False` guard должен быть потолком, а не немедленным удалением старого trace

- Решение: не удалять существующий `active_behavior_allowed=False` из `apply_semantic_frame_proof_reconciliation_shadow`; тест Э0 фиксирует текущий единственный runtime sentinel как allowlist и запрещает новые.
- Обоснование: этот trace уже принят как замороженная будущая основа, тесты отчётов ожидают поле `active_behavior_allowed`. Удалять или переименовывать его в рамках Э0 значило бы менять другой трек, а не резать regex-лазанью.
- Сырьё: `rg active_behavior_allowed` показывает один runtime sentinel в `provider.py:3127`; остальные совпадения — тесты/отчёты.
- Аудит: PASS, это явная правка по замечанию аудитора: делать ratchet/allowlist для старого sentinel, а не ломать уже принятый shadow trace.

### D6. `semantic_reading` не экспортирует P0-поля и не пишет LLM-слоты как подтверждённые

- Решение: новый `semantic_reading.py` читает `model_intent`/`semantic_frame`, но не выносит `risk_class` и `must_handoff` в dataclass. Слоты grade/subject/format выдаются только как кандидаты `source_name="semantic_reading_llm"`, при `source="inline"`, confidence ≥0.70 и детерминированной проверке, что значение звучало в истории.
- Обоснование: текущий этап не имеет права заменять P0/floor/preblock и не должен превращать вывод модели в `client_confirmed`. Это сохраняет будущую пользу для N−1 слотов без загрязнения памяти и без клиентского поведения.
- Сырьё: исходное ТЗ §6.1/П1c; тесты `tests/test_semantic_reading.py` проверяют отсутствие `risk_class`/`must_handoff`, source `semantic_reading_llm`, историю и отсутствие `last_semantic_reading` в prompt-view.
- Аудит: PASS_WITH_NOTES; аудитор отдельно запретил `client_confirmed`, `memory_provenance` confidence 1.0, `child_name` и любые байпасы safety-gates.

### D7. Э2-отчёт расширяется без изменения acceptance-gate

- Решение: добавить в `scripts/report_adr003_semantic_frame_eval.py` опциональный вход `--posthoc-transcripts` и секции `inline_vs_posthoc_agreement` / `baseline_vs_inline_text_health`, но не делать их автоматическим GO на включение.
- Обоснование: Э2 должен дать Claude #1 сырьё для регрейда inline-vs-posthoc и health baseline-vs-inline. Автоматическое разрешение поведения на основании метрик было бы преждевременным: по ТЗ semantic review всех изменённых B↔I ходов делает Claude #1.
- Сырьё: тест `test_report_compares_inline_with_posthoc_and_text_health` проверяет mismatch по `requested_action`; старые report-тесты сохраняют совместимость без posthoc-входа.
- Аудит: PASS, аудитор подтвердил, что Э2 должен расширять только report/script и не запускать M1/live в этом заходе.

### D8. Pre-push/CI guard не ставится как live hook в этом заходе

- Решение: не писать в `.git/hooks/pre-push` и не добавлять новый GitHub Actions workflow в рамках этого изменения. Runtime-защита реализована pytest-guard'ом и snapshot/budget тестами; hook/CI оформить отдельным ops-ТЗ, если Дмитрий решит включить это в общий процесс пуша.
- Обоснование: настоящий git hook — локальная неотслеживаемая мутация рабочего окружения; новый CI workflow меняет инфраструктурное поведение всего репозитория. Это выходит за безопасные Э0/Э1/Э2 изменения и не нужно для локальной приёмки текущего кода.
- Сырьё: в репозитории нет `.github/workflows`; ТЗ само говорит, что pytest не видит коммитную историю, значит commit-level check должен быть отдельным инфраструктурным слоем.
- Аудит: ожидает финального подтверждения; решение выбрано как более безопасное, потому что не меняет процесс пуша без отдельного согласия.

### D9. Слотный floor усилен по словарю П1c

- Решение: использовать закрытые словари grade/subject/format и жёсткую floor-сверку с клиент-авторскими репликами. Для grade нельзя считать поддержку по простой подстроке цифры; даты, мульти-дети и переходные формулировки должны давать `slot_write=none`.
- Обоснование: файл `2026-07-03_slot_dictionary_p1c_v1.json` и разметка 19 строк показывают, что confidence не спасает от KB-копирования и случайных цифр; единственная безопасная защита — проверка, что значение реально было сказано клиентом в правильном контексте.
- Сырьё: добавлены тесты на `6-17 июля`, `9 и 7 класс`, `6-й закончил`, `1го класса`, `программирование`, `из дома`.
- Аудит: ожидает финального подтверждения; решение снижает риск загрязнения памяти и не меняет runtime-поведение при default-OFF масках.

### D10. Новые Foton сценарии остаются входом для Э3/M1, не копируются в runtime

- Решение: не копировать `2026-07-03_scenarios_paket1_neg_pos_personas.jsonl` в runtime-код текущего Э0/Э1. Использовать его как будущий M1/eval вход для Э3-A после регрейда Э2 и отдельного решения Дмитрия.
- Обоснование: текущий заход не включает маски классов и не запускает М1. Сценарии полезны, но их включение в репо без готового runner/manifest сейчас создало бы видимость завершённой приёмки.
- Сырьё: сценарии содержат 13 POS/NEG персон по sense seats, off_topic и slots; это ровно материал для последующего пакетного замера, а не для default-OFF foundation.
- Аудит: ожидает финального подтверждения.

### D11. Слотный floor не считает адрес/площадку форматом "очно"

- Решение: убрать `адрес`/`площадка` из alias-доказательств формата `очно` в `semantic_reading.py`; принимать только реальные форматные слова (`очно`, `очный`, `офлайн`, `в центр`, `в центре`, `приезжать`).
- Обоснование: адресный вопрос должен оставаться адресным вопросом. Иначе мы снова получаем старую ошибку понимания: модель/словарь превращает "где находится площадка?" в подтверждённый формат `очно`.
- Сырьё: внешний словарь П1c разрешает `в центре/в центр/приезжать`, но не `адрес/площадка`; аудит отметил риск `в центре -> очно`, а `адрес -> очно` ещё шире и опаснее.
- Аудит: ожидает финального подтверждения; добавлен NEG-тест на `адрес`/`площадка`.

### D12. Не писать LLM-слоты при выборе или мульти-значении

- Решение: если клиент говорит два предмета или выбор формата (`математика и физика`, `очно или онлайн`), `slot_candidates_from_reading` не пишет одиночный `subject`/`format`, даже если модель выбрала одно значение.
- Обоснование: одиночный слот персистентен. Записать одно значение из выбора означает притвориться, что клиент уже выбрал. Это загрязняет анти-переспрос и будущую память.
- Сырьё: П1c-словарь прямо требует reject для `математика, физика` и `очно или онлайн`; разметка slot-grade показывает тот же класс ошибок для неоднозначных классов.
- Аудит: ожидает финального подтверждения; добавлены NEG-тесты на multi-subject и format-choice.

### D13. `last_semantic_reading` сохраняется в state, но не в prompt-view

- Решение: реальный draft loop передаёт `SemanticReading.from_result(result)` в `update_dialogue_memory_after_answer`; `last_semantic_reading` сохраняется в JSON state и восстанавливается, но не попадает в `to_prompt_view()`.
- Обоснование: П1c требует память хода N-1 для будущих масок, но Э0/Э1 не имеют права менять ответ бота. Скрытое state-поле даёт сырьё для Э3, не подсказывает модели "клиент подтвердил X" и не включает анти-переспрос.
- Сырьё: `draft_loop.py` — единственный реальный Wappi-путь обновления dialogue memory; без прокидывания из draft loop поле оставалось бы тестовой заглушкой.
- Аудит: ожидает финального подтверждения; добавлен тест на сохранение в `state.json` и отсутствие в prompt-view.

### D14. `off_topic` в model_intent является metadata-only до Э3

- Решение: `off_topic` остаётся допустимым значением парсера/payload, но `_intent_model_led_decision` и `_conversation_intent_plan_with_model_led` не используют его для изменения active plan; trace пишет `skip_reason="off_topic_metadata_only"`.
- Обоснование: аудитор нашёл блокер: если просто добавить `off_topic` в allowed-list, уже включённый `intent_model_led` мог демоутить `live_availability/schedule/address/...` в `general_consultation` до включения semantic-reading маски. Это нарушало default-OFF и меняло route/text в Э1.
- Сырьё: `INTENT_MODEL_LED` default-ON под `pilot_gold_v1`; `_conversation_intent_plan_with_model_led` раньше применял любой allowed non-target как `general_consultation`.
- Аудит: BLOCKER принят; добавлен регрессионный тест `test_intent_model_led_off_topic_is_metadata_only_until_semantic_reading_class_enabled`.

### D15. Slot-floor принимает только клиент-авторские строки

- Решение: `slot_candidates_from_reading` игнорирует строки с префиксами `Ответ:/Бот:/bot:/assistant:` при floor-сверке и корректно очищает `Клиент:/user:/client:`; также ловит comma/slash multi-choice.
- Обоснование: future caller может передать смешанную историю, а не уже отфильтрованные реплики клиента. Если floor поверит словам бота, он снова сможет записать KB/ответный факт в память клиента.
- Сырьё: текущая функция принимает `history_texts` как свободную последовательность строк; Wappi history часто сериализуется с префиксами ролей.
- Аудит: RISK принят; добавлены тесты на bot/client-префиксы, `математика, физика` и `очно/онлайн`.

### D16. Ш1 продолжения исполняется без переключения use-site

- Решение: в заходе по `2026-07-03_TZ_Sh1_Sh4_semantic_reading_prodolzhenie_i_prompt_D1.md` делать только безопасный Ш1 и подготовку Ш2-отчёта: floor-фиксы, frozen-guard tests, чистые reader-функции и offline-agreement в report. Ш3/Ш4 не выполнять.
- Обоснование: Ш3 включает активные маски `sense_seats/off_topic/slots_gsf`, а Ш4 удаляет legacy-потребителей. Оба шага требуют регрейда тройки B/I/P и отдельных решений Дмитрия.
- Сырьё: Foton-ТЗ прямо разделяет Ш1/Ш2/Ш3/Ш4 и запрещает use-site переключение в Ш1.
- Аудит: PASS; субагент-аудитор подтвердил, что pure reader functions безопасны только как offline/report.

### D17. `gold-19` используется как ratchet с явными known gaps, а не как абсолютная истина

- Решение: добавить `tests/fixtures/adr003_slot_gold_19_machine_readable.json`, но тестировать его с явным списком спорных строк: `wappi_pair_missing_72h_004`, `wappi_pair_missing_72h_012`, `wappi_pair_missing_72h_019`, `wappi_pair_missing_72h_020`.
- Обоснование: эти строки либо не содержат в `client_quotes` явной клиентской фразы с классом, либо требуют принять `закончил N класс` как текущий класс. Подгонять floor под такие строки опасно: это снова позволит записывать KB/пересказ как клиентское подтверждение.
- Сырьё: `2026-07-03_slot_gold_19_machine_readable.json` хранит данные как объект `.rows`; 2 строки unresolved. Прогон floor показал, что все NEG остаются `none`, а known gaps находятся только среди POS.
- Аудит: PASS_WITH_NOTES; аудитор заранее предупредил, что в gold-19 есть спорные POS и правило переходных фраз требует уточнения Claude #1.

### D18. `history/persona` не является клиентским доказательством без явного `Клиент:`

- Решение: floor теперь извлекает client-authored segments: `Клиент:/user:/client:/turn_msg:` принимаются, `Ответ:/Бот:/assistant:` игнорируются, `history/persona:` принимается только если внутри есть явный клиентский сегмент `Клиент:/client:`.
- Обоснование: enriched/gold строки смешивают текущую реплику, историю и KB/ответы. Без role parsing строка с KB-фактом `для 5-10 классов` могла записать `5` как клиентский слот.
- Сырьё: тест `test_slot_gold_19_floor_has_no_false_writes_and_known_fixture_gaps_are_explicit`; до фикса строка `wappi_pair_missing_72h_003` ошибочно давала `slot_write=yes`.
- Аудит: принято как усиление D15.

### D19. Multi-number grade guard различает класс и цену

- Решение: `_history_supports_grade` больше не режет сообщение из-за любой второй цифры 1-11 рядом со словом `класс`; неоднозначностью считаются только несколько явных grade-фраз. `8 класс, стоимость 9 000 ₽` даёт grade=8, но не grade=9.
- Обоснование: прежний guard безопасно терял recall на легитимном классе, если рядом была цена. При этом защита от multi-child сохраняется через несколько явных grade-кандидатов.
- Сырьё: тесты `test_slot_candidates_accept_grade_near_non_class_price_number`, `test_slot_candidates_reject_grade_from_dates_multi_children_and_transitions`.
- Аудит: PASS; это был основной дефект Ш1.

### D20. Reader agreement добавлен как отчёт, не как политика

- Решение: `scripts/report_adr003_semantic_frame_eval.py` считает `reader_agreement` между legacy-детекторами и pure readers (`sense_seats_reading_decision`, `off_topic_reading_decision`, `slots_reading_candidates`) на inline-транскриптах.
- Обоснование: Ш2 должен доказать готовность к переключению читателей без новых M1-прогонов и без включения масок. Отчёт показывает расхождения, но не меняет route/text.
- Сырьё: `test_report_includes_reader_agreement_for_pure_semantic_readers`.
- Аудит: PASS; аудитор указал, что текущий report умел inline-vs-posthoc, но не умел offline-agreement чистых читателей.

### D21. `draft_loop` запись `last_semantic_reading` не считать безусловной

- Решение: уточнить формулировку ТЗ/отчёта: `draft_loop.py` пишет `last_semantic_reading` только внутри ветки `_memory_provenance_enabled()`.
- Обоснование: Foton-ТЗ называл запись безусловной. Это неточно и может привести к неверному выводу о runtime-памяти.
- Сырьё: `src/mango_mvp/integrations/draft_loop.py` строки вокруг 908-922: semantic reading формируется и передаётся в `update_dialogue_memory_after_answer` только в блоке memory provenance.
- Аудит: PASS; замечание аудитора принято.

### D22. Э2 получает новый merged-набор, а не правку старого канона

- Решение: добавить отдельный сценарный файл `product_data/telegram_dynamic_test_sets/adr003_semantic_reading_paket1_e2_20260703.jsonl`: 2 spec-строки из P1C-файла Foton, 131 persona из старого канона и 15 новых P1C-persona.
- Обоснование: старый `adr003_semantic_frame_m1_scenarios_20260701.jsonl` содержит `shadow_changed_behavior` как hard gate судьи. Новый P1C judge_spec правильно переносит это в soft `shadow_behavior_note`, потому что одна нога не видит paired baseline. Перезаписывать старый канон нельзя: это исторический источник прошлых M1-прогонов.
- Сырьё: новый набор загружается `load_dynamic_sim_input`, содержит `146` persona, без duplicate `dialog_id`; `shadow_changed_behavior` отсутствует, `shadow_behavior_note` присутствует только в `soft_flags`.
- Аудит: PASS; это закрывает замечание аудитора, что сценарный файл без spec-строк не запустится.

### D23. Э2 runner фиксирует `PYTHONPATH` и пустые semantic-reading маски

- Решение: добавить `scripts/run_adr003_semantic_reading_e2_triple.sh` для ручного M1/локального запуска B/I/P. Скрипт выставляет `PYTHONPATH=$ROOT/src`, чистит semantic-frame env на baseline/posthoc ногах и оставляет `TELEGRAM_SEMANTIC_READING_CLASSES=` пустым.
- Обоснование: без явного `PYTHONPATH` Python может импортировать соседний старый checkout из `Mango analyse`, что уже воспроизвелось при вызове `--help`. Ш2 должен проверять telemetry/agreement, а не случайную версию кода или активные маски.
- Сырьё: `bash -n scripts/run_adr003_semantic_reading_e2_triple.sh`; `run_telegram_dynamic_client_sim.py --help` проходит только с `PYTHONPATH=src`.
- Аудит: PASS; runner не запускается автоматически и не трогает live/profile/P0.

### D24. Э2 runner обязан падать, если замер не на direct path

- Решение: после невалидной M1-тройки runner включает `pilot_gold_v1`/direct-path env явно и проверяет B/I ноги до отчёта: профиль активен, `bot_direct_draft > 0`, `bot_direct_path` есть на всех ходах, а в I-ноге inline `bot_semantic_frame` с `source=inline` есть минимум на 99% ходов. Добавлен `--dry-check` на 2 диалога и `sha_manifest.json` с хэшами сценария/snapshot/runner.
- Обоснование: M1-прогон `95b968f4` был `measurement_bug`: профиль был выключен, `bot_direct_draft=0`, `bot_direct_path=0`, `bot_semantic_frame=0`, но отчёт всё равно строился. Такой прогон не отвечает на вопрос Э2 и должен падать инфраструктурно.
- Сырьё: `2026-07-03_REGRADE_troika_NEVALIDNA_fix_i_prompt_D1.md`; поля старого `dynamic_summary.json`/транскриптов: `profile.effective=false`, `bot_direct_draft=0`, `direct=0`, `frames=0`.
- Аудит: встроены статические тесты runner-а; смысловой регрейд новой тройки остаётся за Claude/Fable по сырым M1-логам.

### D25. Frame-emission мерить только на eligible model-called ходах

- Решение: заменить all-turn порог `bot_semantic_frame` на `eligible_frame_rate = frames / eligible_model_called_turns`. Из знаменателя исключаются только два класса: direct-path P0-preblock до модели (`model_called=false`, `preblocked=true`, `preblock_reason in {p0_pre_gate, direct_path_preblocked_p0}`) и `provider_error=timeout`. Прочие provider/fallback-состояния остаются в знаменателе.
- Обоснование: M1-прогон `7676a902` показал валидный direct path, но часть P0-ходов завершается deterministic floor до LLM, поэтому inline frame там невозможен по конструкции. Считать такие ходы как frame-miss — ошибка измерителя. Timeout тоже не является качеством frame, но должен оставаться явной infra-меткой, а не исчезать.
- Сырьё: в `I`-ноге прогона `adr003_semantic_reading_e2_7676a902_20260703_171759` было `269` ходов, `202` inline-frame, `66` direct-path P0-preblock и `1` timeout; на eligible-ходах frame-emission = `202/202`.
- Аудит: `reader_agreement` считается только на ходах, где есть frame/model-intent; это остаётся верным. Runner/report теперь печатают `turns_total`, `preblocked_p0`, `timeouts`, `model_called_eligible`, `frames`, `eligible_frame_rate`; `sha_manifest.json` дополнительно фиксирует хэши B/I/P/REPORT-артефактов.
