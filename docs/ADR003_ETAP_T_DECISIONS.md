# ADR003 этап T — журнал решений Codex

Дата: 2026-07-03
Ветка: `codex/adr003-semanticframe-migration`
Стартовый HEAD: `7676a902bd0ac0dc55dc5df18e461565e22339cb`

## D-001. `PROJECT_NOW.md` не коммитить

Решение: обновлять `docs/PROJECT_NOW.md` только локально через `python3 scripts/project_now.py`; exact HEAD и ссылки на артефакты фиксировать в этом журнале и audit pack.

Обоснование: `docs/PROJECT_NOW.md` является generated/ignored файлом по `AGENTS.md` и `.gitignore`. Коммит такого файла нарушил бы проектную дисциплину и смешал runtime-снимок с кодовым изменением.

## D-002. Делать repo-wrapper ТЗ

Решение: создать tracked wrapper в `tasks/_inbox_codex/`, затем перевести его через `scripts/task_move.py --take` и запускать `scripts/preflight.py`.

Обоснование: исходное ТЗ лежит в Foton и не имеет машиночитаемой шапки. Для крупной реализации нужен штатный preflight, иначе границы зон и безопасный collect-only не проверяются.

## D-003. Сузить реализацию до этапа T

Решение: реализовывать только trace, `sense_seats`, `off_topic`, `slots_gsf`, env-matrix, `e3_paired` runner и deletion manifest. Fix1b/Fix2/slots_reask/deal_action не реализовывать.

Обоснование: эти блоки относятся к другим подсистемам и сделают будущий замер неатрибутируемым. `deal_action` уже включён профилем, `slots_reask` требует передачи semantic reading в симуляторную память, Fix1b/Fix2 требуют отдельных смысловых критериев.

## D-004. Trace финализировать после `apply_semantic_frame_decision_shadow`

Решение: добавить `apply_semantic_reading_trace_finalize(...)` после текущего последнего direct-path shadow слоя; старый `frame_decision_shadow` не переименовывать и не менять.

Обоснование: иначе trace может не увидеть итоговые изменения и может сломать существующие отчёты, которые читают старый ключ.

## D-005. `sense_seats` зависит от reliable step1

Решение: если маска `sense_seats` включена, но `TELEGRAM_RELIABLE_ANSWERER_STEP1` выключен, писать trace `suppressed(reason="reliable_step1_off")`; в `e3_paired` включать reliable step1 в обоих плечах.

Обоснование: `apply_reliable_answerer_output_guard` не работает при выключенном reliable step1. Без явной suppressed-записи класс молча стал бы инертным.

## D-006. `semantic_reading_slots` — hidden storage без читателей

Решение: хранить модельные slot-кандидаты отдельно от `known_slots`, `client_confirmed_slots`, `topic_focus`, prompt-view и direct-path prompt. В этом этапе нет потребителя этих слотов.

Обоснование: модельная догадка не является подтверждённым словом клиента. Любой merge в общий slot-map создаст старую ошибку: бот перестанет переспросить неподтверждённые данные.

## D-007. Маска `slots_gsf` в memory-layer

Решение: не добавлять новый параметр `context` в `update_dialogue_memory_after_answer`; гейтить запись `semantic_reading_slots` через существующий env/context resolver `reading_class_enabled(None, "slots_gsf")`, а caller-side включение оставить future work.

Обоснование: у функции уже есть параметр `semantic_reading`, но нет `context`. Добавление `context` расширило бы публичный контракт памяти и затронуло бы больше вызовов. Env-only гейт достаточен для этого этапа, потому что `TELEGRAM_SEMANTIC_READING_CLASSES` уже является глобальным флагом reading-классов.

## D-008. `e2_triple` не трогать

Решение: не менять `scripts/run_adr003_semantic_reading_e2_triple.sh`; для будущего ON-замера создать отдельный `scripts/run_adr003_semantic_reading_e3_paired.sh`.

Обоснование: `e2_triple` нужен для теневой проверки inline/posthoc frame и специально держит `TELEGRAM_SEMANTIC_READING_CLASSES=` пустым. Смешивать его с active reading-масками нельзя.

## D-009. Deletion manifest — только данные

Решение: создать `docs/ADR003_DELETION_MANIFEST.md` со статусом `awaiting_green`; ничего не удалять.

Обоснование: удаление legacy regex возможно только после per-class зелёного замера и отдельного разрешения Дмитрия.

## D-010. E3 мерит reading-дельту поверх одинакового inline SemanticFrame

Решение: в `e3_paired` включить `TELEGRAM_SEMANTIC_FRAME_SHADOW=1` в обоих плечах, а ON-плечо отличается от B-плеча только `TELEGRAM_SEMANTIC_READING_CLASSES`.

Обоснование: `slots_gsf` читает `requested_product` из inline SemanticFrame. Если frame отсутствует в обоих плечах, paired-замер не измеряет reading-класс. Это не меняет runtime default/profilе и не включает флаги в живом боте; это только решение измерительного runner.

## D-011. Локальный timeout E3 dry-check не блокирует кодовый этап

Решение: локальный E3 `--dry-check`, остановившийся на `provider_error=timeout` в первых двух диалогах, классифицировать как `infrastructure_bug`. Для закрытия кодового этапа добавить unit-инвариант fake direct-provider: при всех reading-масках OFF `semantic_reading_trace` отсутствует, а metadata/route/text остаются исходными. Поведенческий E3 dry-check остаётся обязательным fail-fast предусловием будущего M1/E3 прогона.

Обоснование: timeout случился до появления inline SemanticFrame и не доказывает дефект trace/vrezka-кода. Unit-инвариант закрывает красную строку OFF без модельных вызовов, а runner сохраняет поведенческий gate там, где доступен валидный model-run.

## D-012. `sense_seats` не снимает floor на обещание наличия мест

Решение: `sense_seats=not_seats` больше не пропускает deterministic `availability_promise` guard, если в ответе уже есть обещание наличия места/группы/записи. В таком случае старый floor остаётся, а trace пишет `reason="availability_promise_floor_kept"`.

Обоснование: LLM должна понимать смысл слова «место», но не должна отключать защиту от клиентского обещания «места есть / запишем» без live-факта. Детерминизм здесь остаётся верификатором, а не понимальщиком.

## D-013. PR-A Fix1b пропускает только полностью проверенный клиентский ответ

Решение: `TELEGRAM_FIX1B_AUTONOMY_VERIFIED_FACTS` оставлен default-OFF и не добавлен в профиль. Коридор может снять только два ложных демоута (`autonomy_default_cautious_missing_facts`, `autonomy_default_cautious_unverified_fact`) и только если весь черновик поддержан свежими client-safe фактами, не содержит неподтвержденных чисел/дат, чужого бренда или live-обещаний мест. Trace-класс `fix1b` не входит в `TELEGRAM_SEMANTIC_READING_CLASSES`; это диагностическая запись применения коридора, а не новый пользовательский reading-класс.

Обоснование: найденный баг находится в проверке выхода: бот уже дал проверенный ценовой/адресный ответ, но старый cautious-layer понижал его в `draft_for_manager`. Расширять понимание клиента здесь не нужно. Reader agreement и будущий регрейд должны видеть, где коридор сработал, но P0/live/brand/fabrication полы остаются выше и не обходятся.
