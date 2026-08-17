> DONE 2026-08-17 12:44 | ветка codex/mango-calls-quality-v3-m4-20260816 | codex

> TAKE 2026-08-16 18:36 | ветка codex/mango-calls-quality-v3-m4-20260816 | codex

Ветка: codex/mango-calls-quality-v3-m4-20260816
Зоны: src/mango_mvp/models.py, src/mango_mvp/db.py, src/mango_mvp/cli.py, src/mango_mvp/clients/ollama.py, src/mango_mvp/services/dialogue_contract.py, src/mango_mvp/services/analyze.py, src/mango_mvp/services/resolve.py, src/mango_mvp/services/transcribe.py, src/mango_mvp/services/ingest.py, src/mango_mvp/services/export_excel.py, src/mango_mvp/services/export_ai_office.py, src/mango_mvp/services/sync_amocrm.py, src/mango_mvp/services/llm_response_cache.py, src/mango_mvp/amocrm_runtime/deal_dossier.py, src/mango_mvp/amocrm_runtime/deals.py, src/mango_mvp/quality/tenant_text_normalizer.py, src/mango_mvp/quality/crm_writeback_quality_detector.py, src/mango_mvp/productization/mango_office_client.py, src/mango_mvp/productization/capture_staging.py, src/mango_mvp/customer_timeline/calls_two_processes.py, src/mango_mvp/customer_timeline/canonical_readonly_import.py, src/mango_mvp/customer_timeline/objections.py, src/mango_mvp/customer_timeline/manager_dossier.py, src/mango_mvp/customer_profile/builder.py, src/mango_mvp/maintenance/canonical_master.py, scripts/build_mango_call_timeline_increment.py, scripts/export_daily_mango_calls_resolve.py, scripts/publish_live_mango_calls_google.py, scripts/publish_current_mango_calls_google.py, scripts/run_mango_calls_publication_coordinator.py, deploy/mango_calls_live_publisher/, tests/, docs/, tasks/, audits/_inbox/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_dialogue_contract.py tests/test_analysis_schema.py tests/test_analyze.py tests/test_analyze_xa_safe_pack.py tests/test_controlled_call_scope.py tests/test_publish_live_mango_calls_google.py tests/test_publish_current_mango_calls_google.py tests/test_mango_calls_publication_coordinator.py tests/test_export_daily_mango_calls_resolve.py tests/test_dialogue_format.py tests/test_tenant_text_normalizer.py tests/test_crm_writeback_quality_detector.py tests/test_resolve.py tests/test_mango_office_client.py tests/test_productization_capture_staging.py tests/test_export_excel.py tests/test_ai_office_export.py tests/test_amocrm_deals.py tests/test_legacy_amocrm_sync_guard.py tests/test_tallanto_export.py tests/test_llm_response_cache_contract.py tests/test_ingest_filename_parse.py tests/test_mango_call_timeline_increment.py tests/test_mango_calls_two_processes.py tests/test_mango_calls_m1_bootstrap.py tests/test_canonical_master.py tests/test_customer_timeline_canonical_readonly_import.py tests/test_customer_timeline_manager_dossier.py tests/test_customer_timeline_objections.py tests/test_customer_profile_builder.py
Семантический-аудит: да

# ТЗ: связный выпуск качества Mango Calls v3 для передачи на M1

Дата: 2026-08-16.
База: `c18ca4c830b9ac122363b2369567add37a11ca8d`.

## 1. Цель простыми словами

Довести существующий конвейер звонков до состояния, где:

1. Analyse получает полный разговор по порядку реплик, а не два монолита дорожек.
2. Система называет стороны «Менеджер/Клиент» только по подтверждённой разметке дорожек Mango.
3. При сомнении сохраняются «Спикер A/B», звонок не теряется и требует проверки.
4. Рискованные выводы можно проверить по конкретной реплике и таймкоду.
5. Время единообразно показывается по Москве, а известные ошибки ASR исправляются только детерминированными правилами без изменения сырья.
6. Google-выгрузчик не скрывает ошибки и не вызывает Codex.
7. Неизменный звонок не вызывает модель повторно; точный расход токенов сохраняется, если его сообщил провайдер.
8. Первый запуск нового prompt-контракта считается холодным: массовая обработка запрещена без измерения и бюджета на 50 звонках.

Первый выпуск включает только код, синтетические/локальные тесты и пакет передачи. Реальные Mango API, ASR, Resolve, Analyse, Google, launchd, M1 runtime, AMO, Tallanto и Customer Timeline не запускаются и не изменяются.

## 2. Источники требований

- `docs/mango_calls_handoff_20260816/TZ-01-canonical-dialogue-input.md`;
- `TZ-02-role-attribution-guard.md`;
- `TZ-03-claim-evidence.md`;
- `TZ-04-safe-normalization-msk.md`;
- `TZ-05-versioned-google-publisher.md`;
- решения D-117 и D-118 в `docs/DECISIONS_LOG.md`;
- замечания владельца от 2026-08-16 о дорожках Mango и экономии токенов.

Это ТЗ задаёт единственный исполнимый порядок и отменяет противоречащие пункты старых документов.

## 3. Явные решения и отмены

1. `scripts/publish_live_mango_calls_google.py` остаётся единственным Google-выгрузчиком. Второй не создаётся.
2. Публикационное состояние и инциденты остаются в существующем owner-only JSON sidecar. Таблицы публикации в рабочей SQLite не создаются.
3. `RESOLVE_LLM_PROVIDER=off` действует во всех режимах, включая `controlled_1`.
4. Повторный Analyse защищает существующий `LLMResponseCache`, включённый по умолчанию. Второй флаг и второй кэш не создаются. Каждая внешняя попытка сначала резервируется в существующем `analysis_attempts_json`; кэш заполняется только после подтверждённого commit результата.
5. Только техническая разметка сторон, полученная от Mango, является основанием ролей. Официальный API `recording_transcripts` возвращает хронологические фразы с метками `client/operator`; capture должен сохранять этот ответ как неизменяемое provider evidence. Смысл текста, приветствие, направление, имя файла и порядок каналов не используются для угадывания.
6. Raw audio, оба ASR-результата, исходные реплики и Resolve evidence неизменяемы.
7. Смена/понижение модели не входит в работу и допустима только после слепого корпуса.
8. Не выполнять требование «до пяти аудитов после каждого файла»: один аудит Claude на связный этап и один финальный сквозной; повтор только для нового P0/P1.

## 4. Бритва и бюджет

Рассмотрены варианты:

1. Исправить только выпадение нейтральных ролей: мало, оставляет неверный вход Analyse.
2. Минимально расширить существующие Analyse и publisher общим контрактом диалога: выбран.
3. Переписать конвейер и publisher: отклонён как рискованный дубль.

Разрешён один новый production-модуль: `dialogue_contract.py`, потому что один и тот же контракт нужен Analyse и publisher. Новых зависимостей, БД, таблиц и параллельных путей: 0. Каждый из четырёх этапов ниже является отдельным механизмом с пределом до 150 добавленных строк нетестового кода; превышение требует остановки и упрощения, а не скрытого разрастания.

## 5. Этап A: общий контракт, нейтральные роли и видимые ошибки

1. Сначала создать один компактный parser/projection в `dialogue_contract.py`, затем использовать его в Analyse, Resolve и publisher. Второй локальный parser не добавлять.
2. Исправить `role_name`/`render_transcript`: корректные mono, neutral-speaker, transfer/conference/echo и неподтверждённые stereo публикуются как `Спикер A/B`, без обратного восстановления ролей.
3. Любая невалидная строка создаёт обезличенный устойчивый incident в текущем sidecar.
4. Отчёт содержит `health.status=green|amber|red`, число открытых инцидентов и нарушение SLA.
5. Открытая ошибка проекции не мешает доказанно валидным соседним строкам, но exit code не равен нулю. Ошибка идентичности блокирует write/sort/layout всего запуска.
6. Подготовить только шаблон прямого launchd-запуска publisher без Codex; не устанавливать.

Приёмка:

- 8/8 синтетических топологий не теряются;
- при недоверенной роли нет меток «Менеджер/Клиент»;
- одна плохая и одна хорошая строка дают одну опубликованную строку, один incident и незелёный статус;
- повтор ошибки сохраняет `first_seen_at`, не плодит occurrence и не содержит ПДн;
- второй publisher и новая SQLite-схема отсутствуют.

## 6. Этап B: канонический диалог и защита ролей

Создать `DialogueInput v1` из сохранённых `transcript_variants_json.dialogue_lines`:

- стабильные `turn_id=T0001...`;
- порядок реплик без перестановки;
- `start_sec`, `timecode`, физический канал, безопасное отображаемое имя и полный текст;
- SHA канонического диалога;
- `role_attribution` с `trusted|untrusted|not_applicable` и закрытыми reason codes;
- явный `transcript_text_fallback`, который всегда требует проверки.

Trusted разрешён только при валидном provider evidence Mango и однозначном соответствии его `client/operator` физическим каналам/репликам. Существующий `status=confirmed_multi_signal`, построенный по словам разговора, сам по себе **не доверенный**. Пока provider evidence не получен или соответствие неоднозначно, используются только `Спикер A/B`. Mono/transfer/conference/echo/corrupt/model correction/missing mapping всегда untrusted. Доказанный non-conversation может быть not_applicable.

В `MangoOfficeClient` добавить только чистый метод пакетного чтения `/vpbx/queries/recording_transcripts` и строгий parser ответа. Live-вызов в этой задаче запрещён. Фактическая доступность метода и связывание на M1 проверяются лестницей 1→10 до content cutover.

Analyse получает отрендеренные целые реплики с `turn_id`; сокращение выполняется только по границам реплик. При untrusted:

- `needs_review=true`, причина `role_attribution_untrusted`;
- нет role-dependent следующего шага, срока, контактов, ФИО и утверждений «клиент/менеджер сделал»;
- нейтральная тема может сохраниться только как нейтральная тема разговора.

Приёмка:

- детерминированный SHA и порядок;
- повреждённая реплика не исчезает тихо;
- одинаковые таймкоды не меняют порядок;
- 0 role-dependent полей у untrusted;
- потеря lease или изменение входного SHA до commit отбрасывает устаревший ответ.

## 7. Этап C: доказательства, нормализация и время

1. Для рискованных непустых выводов хранить `claim_evidence`: путь поля, тип подтверждения, ссылки на существующие turn_id, точный текст и таймкод, собранные сервисом из DialogueInput.
2. Модель может запросить ссылки только через закрытый список полей; она не задаёт цитату, SHA и окончательный claim_id.
3. Несуществующая реплика, inferred вместо explicit, противоречие отрицанию или неподтверждённая роль делают поле невалидным и отправляют в review.
4. Итоговый управленческий конспект строится только из подтверждённых значений; свободный модельный текст не становится доказанным фактом.
5. Расширить существующий tenant normalizer provenance-обёрткой с `rule_id`, `engine_version`, `ruleset_version`, `tenant_id`, не создавая второй словарь и не меняя raw.
6. Использовать одну `Europe/Moscow`-функцию. Naive datetime трактуется как UTC. Убрать из конспекта дубли даты и имени менеджера.

Приёмка:

- 100% опубликованных high-risk полей имеют valid explicit turn evidence;
- отсутствующая/чужая ссылка не публикуется;
- точная цитата принадлежит тому же звонку;
- raw SHA до/после совпадает;
- повторная нормализация побайтно одинакова;
- UTC→МСК корректен на переходе суток и года;
- брендовые правила не протекают в другой tenant.

## 8. Этап D: версии и экономия токенов

1. В `analysis_meta` записать `analysis_input_sha256`, версии диалога/ролей/промпта/нормализатора/часового пояса, provider/model и факт cache hit/model call.
2. Точный `token_usage` брать только из ответа провайдера: полный набор помечать `provider_exact`, неполный `provider_partial`, отсутствие `unavailable`. Не оценивать символами. Баланс модельных вызовов должен сходиться по provider/model/prompt_version.
3. Повтор неизменного запроса обязан попасть в существующий кэш и не вызвать модель. Изменение модели, версии промпта или входа меняет ключ.
4. Publisher fingerprint включает версии и SHA результата Analyse; старое verified становится stale по существующему пути.
5. Сформировать закрытый баланс как вычисляемый отчёт поверх существующих `reserved|verified`, инцидентов и текущего run; новую машину состояний не создавать. Каждый `analysis_status=done` находится ровно в одной вычисляемой категории `verified_current|published|reserved|quarantined|failed_with_incident`. Несходящийся баланс блокирует внешний write.
6. Второй неизменный запуск publisher: 0 Google batchUpdate, 0 дублей.
7. Устранить регрессию M1: `worker_environment()` не должен включать `RESOLVE_LLM_PROVIDER=codex_cli`; безопасное значение `off` обязательно и для controlled-профиля. Отдельно показать счётчик вызовов Resolve LLM = 0.
8. Перед внешним вызовом Analyse записать уникальный `attempt_id` и состояние `reserved`; после ответа завершить ту же запись. `reserved`/`indeterminate`, повтор ID и повреждённый журнал блокируют Google до первого и каждого компенсационного запроса.
9. `analysis_attempts_json` является источником истины по расходу; вложенный `analysis_meta.model_attempts` — только совместимость со старыми строками. Потерянный commit-ack проверяется обратным чтением, а не повторной записью попытки.

Приёмка:

- 0 повторных модельных вызовов на неизменном prompt при включённом штатном кэше;
- OpenAI usage совпадает с mock-response usage;
- Codex/mock без usage честно помечены unavailable;
- изменение входа/версии вызывает ровно один новый анализ;
- cache put происходит только после подтверждённого `analysis_status=done`;
- потерянный commit-ack оставляет одну попытку и возвращает success;
- незакрытая/дублированная попытка даёт 0 Google batchUpdate;
- 0 stale verified и закрытый баланс на офлайн-фикстуре.

## 8-бис. Поправка после независимого аудита A/B (2026-08-16)

Независимые аудиторы (Claude + Codex) доказали воспроизводимыми сценариями классы
дефектов ниже. Все они входят в объём одного связного ремонта A/B и закрываются
отрицательным тестом каждый.

| Класс | Подтверждённый дефект | Закрывается |
|---|---|---|
| A | Resolve: устаревший worker перезаписывает свежий результат и экспортирует stale-файл | полный input snapshot + условный переход + экспорт только после commit |
| B | provider evidence перепривязывается к другому звонку; `evidence.channels` инвертируется отдельно; raw response не сверяется с `dialogue_lines` | mapping выводится заново из raw-фраз; сверка фраз с `dialogue_lines`; `channels` не является вторым источником |
| C | соседние строки сливаются и теряют собственные `turn_id`/таймкод | одна исходная строка = один turn |
| D | `migrate_analysis_payload` и `export_ai_office` оживляют очищенные поля; `export_excel` создаёт follow-up у недоверенного звонка | общий fail-closed allowlist во всех трёх точках |
| E | exception-путь Analyse крадёт чужой lease; prompt metadata не входит в stale guard | условный UPDATE и на ошибке; prompt identity внутри snapshot |
| F | Resolve имеет второй parser/whitelist; промпт сам предлагает менять speaker/swap | общий `parse_dialogue_lines`; промпт запрещает; runtime guard отклоняет |
| G | publisher публикует старые `analysis_json` без role guard; инциденты duplicate identity/reconcile недолговечны | allowlist к старым payload; классовые устойчивые инциденты |
| H | fallback Analyse игнорирует `full.final` | fallback идёт через канонический render контракта |
| I | усечение по краям не маркируется | маркер головы/середины/хвоста внутри бюджета |
| J | пустой диалог ошибочно `not_applicable` | пустой диалог = `untrusted` |
| K | клиент не соответствует официальному конверту `recording_transcripts` | запрос по `recording_id`, ответ `result` + `data.recording_id/names/phrases` |

### Второй раунд независимого аудита A/B (2026-08-16, вечер)

Аудиторы доказали, что после первого ремонта A/B принимать нельзя: `trusted`
физически недостижим в реальном producer, а untrusted всё равно платит токенами.
Классы ниже входят в тот же связный ремонт, каждый закрыт отрицательным тестом.

| Класс | Подтверждённый дефект | Закрывается |
|---|---|---|
| L (trusted unreachable) | `TranscribeService._build_dialogue_lines()` пишет `Менеджер/Клиент`, когда `manager_quality_allowed=true`, а контракт одновременно требует и физическую сторону, и `manager_quality_allowed=true`; зелёные тесты использовали невозможную комбинацию | producer всегда пишет `Дорожка левая/правая`; старая строка восстанавливает сторону только из сохранённых `manager.physical_channel`/`client.physical_channel` |
| M (useless model call) | Analyse вызывает модель и кэш для untrusted и затем удаляет role-dependent результат: лишние токены плюс риск утечки смысла в соседнее поле | при untrusted модель и кэш не вызываются вообще; payload детерминированный; `token_usage.source=skipped_untrusted_role` |
| N (official schema mismatch) | фикстуры `names: [{name, role, channel}]`, `phrases.start/channel` изобретены нами и «доказывали» привязку дорожек, которой официальный ответ не даёт | `data` — объект одной записи или массив batch-ответа; `names` — объект `{client, operator}`; фраза = пара роль + текст; привязка стороны выводится сравнением текстов; внутренний звонок и одинаковые дорожки — untrusted |
| O (current publisher bypass) | `publish_current_mango_calls_google.py`, его coordinator и `deal_dossier.py` читают старый `analysis_json` без guard и оживляют next_step/срок/возражение/резюме | общий `guard_stored_analysis` во всех трёх точках |
| P (Excel formula) | XlsxWriter компилирует ячейку, начинающуюся с `=`, `+`, `-`, `@`, в формулу | `strings_to_formulas=False` + `write_string`; отрицательный тест смотрит XML, а не значение Python |
| Q (oversize disappearance) | звонок с расшифровкой > лимита Google исчезал из отчёта целиком | строка остаётся: целые реплики в жёстком лимите, видимый маркер пропуска, явная причина проверки; Google не объявляется хранилищем полного текста |
| R (contradictory review) | `Нужна проверка = Нет` при непустой колонке причины | флаг вычисляется из причины: непустая причина всегда даёт «Да» |
| S (hidden artifact failure) | ключ инцидента и поле `call_key` в sidecar содержали сырой `source_call_id`; dry-run терял инциденты; stdout и `last_error` печатали `str(exc)` | ключ и поле — необратимый дайджест; dry-run пишет owner-only sidecar; общий `safe_error_text` (stage/type/hash) |

### Третий раунд независимого аудита (2026-08-17)

Финальные аудиторы нашли соседние пути, которые могли обойти новый контракт.
Каждый класс ниже включён в тот же выпуск и закрыт воспроизводимым тестом.

| Класс | Подтверждённый дефект | Закрывается |
|---|---|---|
| T (evidence rebinding) | независимый `recording_id` жил внутри изменяемого JSON и мог быть перепривязан вместе с evidence | отдельный `call_records.source_recording_id`, перенос из capture manifest через metadata/ingest, сверка guard |
| U (legacy role leak) | stereo-transcribe продолжал заполнять `transcript_manager/client` по эвристике, поэтому старый downstream мог обойти `DialogueInput` | producer хранит только физические дорожки; role-зависимые legacy-поля пусты до доказательства Mango |
| V (late contradiction) | отрицание оплаты или отказ позже ближайших трёх реплик не отменяли ранний положительный вывод | проверка всего последующего диалога и отрицательные тесты позднего отрицания/отмены |
| W (stale downstream) | Timeline, AMO deals и смешанный deal dossier могли прочитать старый `analysis_json` без fail-closed guard | общий `guard_stored_analysis`, запрет LLM/writeback у untrusted, очистка смешанного rollup |
| X (transport PII) | HTTP-ошибка включала тело ответа, а полный пакет Mango копировался в evidence каждого звонка | только status/path/body SHA в ошибке; полный пакет хранится один раз, per-call evidence содержит один конверт |
| Y (pre-write race) | источник мог измениться после сборки Google batch и до внешней записи | повторное чтение всей исходной выборки и точных fingerprint непосредственно перед единственным write |
| Z (false acceptance) | M1 требовал `cache_hit=true` от service-repeat, который вообще не входит в Analyse, и не имел численных критериев пилота | фактический отчёт службы `idle/unchanged_snapshot/new=0/reused=1`, отдельный cache-test Analyse; непересекающиеся выборки и метрики по каждому полю |

### Четвёртый раунд независимого аудита (2026-08-17)

После сквозной проверки смыслового результата и гонок данных подтверждены ещё
несколько соседних классов. Они не закрываются «зелёным pytest» без отдельного
отрицательного сценария.

| Класс | Подтверждённый дефект | Закрывается |
|---|---|---|
| AA (semantic polarity) | условная готовность купить становилась состоявшейся продажей, положительная оценка цены — возражением, уже выполненное действие — будущим шагом | проверка контекста и полярности по всему диалогу + парные положительные/отрицательные тесты |
| AB (prompt truncation) | усечённый вход Analyse мог выглядеть полноценным и не требовать проверки | явный `analyze_prompt_truncated` и обязательный review |
| AC (usage identity) | попытки разных provider/model могли суммироваться под последней моделью | учёт каждой попытки по её собственным provider/model/prompt_version |
| AD (resolve export race) | после commit конкурент мог заменить идентификаторы/путь, а старый worker экспортировал результат под новым именем; reaper мог снять аренду во время экспорта | CAS по трём source-идентификаторам + обновление lease перед экспортом |
| AE (ambiguous basename) | два разных звонка с одинаковым basename могли получить чужой анализ в Timeline | полный путь имеет приоритет; короткий ключ разрешён только при однозначной связи |
| AF (recording duplicate) | два звонка с одним непустым `source_recording_id` могли войти в новый canonical master | fail-closed проверка уникальности до выпуска |
| AG (readonly probe drift) | M1 readiness-probe импортировал удалённые read-only функции Google | восстановлен минимальный read-only контракт и сквозной тест probe |
| AH (compensation boundary) | после трёх успешных компенсаций стабильная четвёртая проверка была недостижима | три компенсации + отдельная финальная стабильная сверка; четвёртое изменение блокирует запись |
| AI (stale deal heuristic) | часть deal-анализа строилась из старого CSV, хотя dossier уже имел свежий защищённый Analyse | heuristic строится из того же защищённого снимка dossier; конфликт закрыт тестом |
| AJ (stale usage loss) | уже оплаченная попытка Analyse исчезала из учёта, если lease или input менялись до записи результата | отдельная source-bound CAS-дозапись только технического ledger; чужие result/lease/status/updated_at неизменны |
| AK (provider attempt gap) | cache put и финальный commit могли разойтись, а потерянный commit-ack создавал повторный учёт | reservation до вызова, finalize того же ID, commit readback и cache-after-commit |
| AL (cost gate after write) | незамкнутый баланс выявлялся только после Google batch; вложенный meta скрывал поздние попытки | DB-ledger имеет приоритет, cost gate выполняется до первого и каждого compensation write |

### Пятый раунд независимого смыслового аудита (2026-08-17)

Каждый подтверждённый пробой ниже закрывается в общем механизме и постоянным
регрессионным тестом. Точечные телефоны, клиенты и тексты не зашиваются.

| Класс | Подтверждённый дефект | Закрывается |
|---|---|---|
| AM (historical/modal polarity) | «прошлым летом» и «хотел бы/можно было бы» становились текущей договорённостью | расширение общего контекстного и модального guard с положительными контролями |
| AN (late outcome reversal) | поздняя отмена звонка, продажи или платежа не всегда отменяла ранний вывод | единые action/result reversal guards по всему последующему диалогу |
| AO (weak refusal object) | «не нужна рассрочка/скидка» ошибочно становилось отказом от продукта | отказ требует явного объекта продажи либо самостоятельного «я отказываюсь» |
| AP (provider token reuse) | одно слово сохранённой дорожки могло одновременно «доказать» две короткие фразы Mango | короткая provider-фраза требует отдельной точной реплики; одна реплика не переиспользуется |
| AQ (false exact usage) | `provider_exact` без полного набора числовых счётчиков считался точным | полный набор неотрицательных целых обязателен, иначе usage только partial |
| AR (hidden Codex retries) | одна задача Analyse могла незаметно вызвать Codex до пяти раз без точного usage | одна зарезервированная попытка = один внешний вызов; повтор только отдельным управляемым запуском |
| AS (runtime cache alias) | разные Ollama endpoint/think/temperature/num_predict могли разделить cache/source identity | хэш всех влияющих runtime-параметров входит в cache key и analysis source identity |

### Исключение из лимита «до 150 добавленных строк на этап»

Лимит §4 нарушен сознательно и зафиксирован здесь честно: пункты A, B, D, E, G —
это fail-closed защиты, а не функциональность. Резать защиту ради строки в бюджете
запрещено: цена ошибки — опубликованный клиенту неверный вывод о том, кто что
сказал, и потерянный результат чужого worker. Поэтому лимит заменяется на два
правила: (1) каждый добавленный блок закрывает доказанный класс дефекта из
таблицы выше и имеет отрицательный тест; (2) сразу после реализации запускается
уборщик, и дубли (второй parser, второй whitelist, второй проекционный путь,
повторяющиеся ветки guard) удаляются. Результат уборки перечисляется в отчёте
отдельной строкой «что упростил».

## 9. Обязательные проверки

1. Целевые unit/integration tests без сети и production runtime.
2. Враждебные случаи: неизвестная роль, mono, transfer, conference, echo, corrupt line, конфликт ASR, чужой turn_id, отрицание оплаты, stale result, crash/retry, формула в тексте.
3. Повтор без изменений и отрицательный контроль каждого нового guard.
4. `git diff --check`, secret scan, полный применимый pytest.
5. Отдельные статусы: `formal_pass`, `semantic_pass`, `business_pass`, `data_pass`, `runtime_pass`.
6. Смысловая выборка для РОПа: ясно ли за 30 секунд, что было в разговоре, что подтверждено, что требует проверки и какой следующий шаг допустим.

Каждая найденная смысловая ошибка превращается в тест, gate или явный ручной контроль.

## 10. Audit pack и передача M1

Создать один `audits/_inbox/mango_calls_quality_v3_<timestamp>/`:

- `implementation_notes.md`;
- `changed_files.txt`;
- `test_output.txt`;
- `semantic_review.md`;
- `risk_review.md`;
- `backward_compatibility.md`;
- `claude_review.md`;
- `requirements_traceability.md`;
- `secret_scan.txt`;
- `manifest.json` последним.

Подготовить `docs/mango_calls_handoff_20260816/M1_QUALITY_V3_INTEGRATION_PROMPT.md` с точной веткой/SHA, командами получения, тестами, проверкой конфигурации и лестницей M1: offline → 1 звонок → 10 → 50 trusted + 50 untrusted → 200 стратифицированных. Ни один live-шаг не исполняется в этой задаче.

## 11. Definition of Done

- код и тесты реализованы в единственном пути;
- P0/P1 независимых аудитов закрыты или доказанно отклонены;
- формальные тесты зелёные;
- смысловой аудит имеет отдельный вердикт;
- branch push выполнен, main не изменён и PR не создан;
- M1 получает точный SHA и исполнимую инструкцию;
- отдельно перечислено, что не доказано без реального пилота;
- production M1, Mango API, Google, рабочие SQLite, ASR, Resolve/Analyze и службы не изменялись.

## СТОП

Немедленно остановить этап и не переходить дальше, если:

- хотя бы одна реплика исчезает без явной причины;
- роль восстановлена без подтверждённого `role_mapping` Mango;
- рискованный факт опубликован без точной ссылки на реплику;
- изменён raw audio, ASR, исходный диалог или Resolve evidence;
- закрытый баланс не сходится либо второй прогон создаёт дубль/модельный вызов/write;
- появляется второй publisher, вторая call DB, новая publication-таблица или дублирующий флаг;
- тест требует production credential, live API, ASR, Resolve/Analyze или внешнюю запись;
- возникает новый P0/P1, который нельзя закрыть воспроизводимым тестом в текущих границах.

## Самодекларация

В итоговом отчёте указать: добавлено/удалено строк; новых production-файлов, флагов и зависимостей; какие более простые варианты рассмотрены и почему отвергнуты.
