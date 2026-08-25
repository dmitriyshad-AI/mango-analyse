> DONE 2026-08-25 12:31 | ветка codex/calls-neutral-analysis-fix-20260824 | codex

> TAKE 2026-08-25 00:49 | ветка codex/calls-neutral-analysis-fix-20260824 | codex

Ветка: codex/calls-neutral-analysis-fix-20260824
Зоны: src/mango_mvp/config.py, src/mango_mvp/cli.py, src/mango_mvp/services/transcribe.py, src/mango_mvp/services/dialogue_contract.py, src/mango_mvp/services/resolve.py, src/mango_mvp/services/analyze.py, src/mango_mvp/customer_timeline/calls_two_processes.py, src/mango_mvp/customer_timeline/manager_dossier.py, src/mango_mvp/customer_timeline/objections.py, scripts/publish_live_mango_calls_google.py, .env.example, tests/test_cli.py, tests/test_mango_calls_two_processes.py, tests/test_dialogue_format.py, tests/test_dialogue_contract.py, tests/test_resolve.py, tests/test_analyze.py, tests/test_export_excel.py, tests/test_ai_office_export.py, tests/test_publish_live_mango_calls_google.py, tests/test_tz118_transcribe_d_primary.py, tests/test_parallel_pipeline.py, tests/test_controlled_call_scope.py, tests/test_codex_merge.py, tests/test_transcribe_suspicious_drops.py, tests/test_customer_timeline_manager_dossier.py, tests/test_customer_timeline_objections.py, docs/mango_calls_handoff_20260825/, docs/worktrees_registry.md, tasks/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_dialogue_format.py tests/test_dialogue_contract.py tests/test_resolve.py tests/test_analyze.py tests/test_export_excel.py tests/test_ai_office_export.py tests/test_publish_live_mango_calls_google.py tests/test_tz118_transcribe_d_primary.py tests/test_parallel_pipeline.py tests/test_controlled_call_scope.py tests/test_codex_merge.py tests/test_transcribe_suspicious_drops.py tests/test_customer_timeline_manager_dossier.py tests/test_customer_timeline_objections.py
Семантический-аудит: да
Feature-ID: feature.calls_stereo_role_orientation
Problem-ID: problem.calls_untrusted_roles_blanket_semantic_refusal
Исход: attempt_complete
Изменение: extend
Ключевые-символы: TranscribeService._classify_stereo_call,TranscribeService._assign_roles_with_codex,evaluate_role_attribution,ResolveService._semantic_selective_input
Ключевые-слова: stereo channel orientation,physical tracks,manager client roles,model role assignment

# ТЗ: определение ролей двух физических дорожек моделью
## Цель

Для обычного двухстороннего звонка один короткий модельный запрос определяет,
какая целая физическая дорожка относится к менеджеру, а какая к клиенту. После
этого работают существующие Resolve и Analyze. Модель не меняет текст и не
назначает роли отдельным репликам.

## Решение

1. Переиспользовать транспорт существующего изолированного вызова Codex
   (очищенное окружение, read-only sandbox, разбор JSON) и `LLMResponseCache`,
   но не его пофразовый mono-промпт и не назначение ролей отдельным репликам.
   Для целых дорожек нужен отдельный построитель запроса.
2. Передавать ему ровно два ограниченных по длине текста: левую и правую
   дорожки. Принимать только две противоположные роли и уверенность не ниже
   `0.85`.
3. Сохранить решение в существующем `role_mapping` со статусом
   `confirmed_model_channel_orientation`; физические дорожки и ASR-текст не
   изменять.
4. Внутренний звонок, конференция, перевод, пустая или дублирующаяся дорожка не
   вызывают модель и остаются неподтверждёнными.
5. Ошибка или неоднозначный ответ модели не теряет звонок: остаётся безопасный
   `Спикер A/B` без оценки менеджера.
6. Модельная ориентация разрешает обычный Analyze, но не открывает выборочное
   модельное редактирование Resolve: этот путь по-прежнему требует независимой
   разметки Mango.
7. Исправить общий дефект: отрицательный, условный или исторический вопрос не
   подтверждается коротким ответом «да».
8. Вызывать модель один раз в `_transcribe_call()` после проверки обеих ASR-
   версий, а не внутри `_classify_stereo_call()`. Расхождение структуры дорожек
   между Whisper и GigaAM сильнее ответа модели и оставляет роли неподтверждёнными.
9. Для ориентации использовать отдельные prompt/version/cache namespace. Из
   модельного ответа принимать только роли двух целых дорожек и confidence;
   сгенерированный текст или реплики игнорировать.
10. В ролевом контракте хранить источник доверия отдельно: модельная ориентация
    открывает Analyze, но только `provider_evidence` открывает selective Resolve.
    Изменение семантики сопровождается новой версией `ROLE_GUARD_VERSION`.
11. `confirmed_multi_signal` остаётся только подсказкой: он больше не выставляет
    `manager_quality_allowed=True` и не открывает Analyze. Analyze открывает
    только `confirmed_model_channel_orientation` либо provider evidence.
12. Совместимая структура двух ASR означает: обе версии сохранили простой
    двухсторонний звонок, в каждой непусты обе дорожки и ни одна версия не
    потребовала mono-fallback. Этот предикат не использует словарный ролевой
    вердикт. При несовместимости модель не вызывается.
13. После принятой модельной ориентации в расшифровке и витринах показываются
    обычные подписи `Менеджер/Клиент`; в метаданных сохраняется источник
    `model_channel_orientation`. При отказе остаются `Спикер A/B`.
14. Принятая модельная ориентация открывает существующие обычные Analyze,
    экспорт, досье менеджера и разбор возражений. Она не открывает только
    selective Resolve, для которого обязательно `provider_evidence`.
15. Без второй ASR-версии модель ориентации не вызывается: звонок остаётся на
    безопасном неподтверждённом пути.

## Настройка выпуска

`STEREO_ROLE_ORIENTATION_MODE=off|codex` по умолчанию `off`.
Владелец: Дмитрий. Точка закрытия флага: после пилота 1→10→25 на сохранённых
ASR-результатах минимум 24 из 25 ролей совпадают с ручной проверкой и нет ни
одной уверенной инверсии. После выполнения критерия M1 включает `codex`; при
неуспехе режим остаётся `off`, примеры идут в доработку.

## Приёмка

1. На простом двухстороннем звонке модель определяет целую дорожку и вызывается
   не более одного раза.
2. Повтор идентичного запроса использует существующий кэш.
3. Перестановка левой и правой дорожки переставляет роли.
4. Две одинаковые роли, низкая уверенность и ошибка модели не подтверждают роли.
5. Внутренний, конференционный, transfer, echo и пустой звонки не вызывают
   модель.
6. `manager_quality_allowed=true` только для принятой простой модельной
   ориентации либо валидной независимой разметки Mango.
7. Resolve не меняет физическую связь и не открывает selective merge без
   provider evidence.
8. Analyze использует прежний доверенный путь; второго анализатора нет.
9. Отрицательный, условный и исторический вопрос + «да» не создают действие.
10. Точечные тесты зелёные; Claude, архитектор, ломатель и бизнес-аудитор не
    находят P0/P1.
11. Ориентация имеет отдельные prompt/version/cache namespace и не использует
    сгенерированный моделью текст.
12. При несовместимой структуре дорожек Whisper/GigaAM модель не вызывается.
13. Новая версия ролевого контракта согласована с экспортом и Google publisher.
14. `confirmed_multi_signal` сам по себе не открывает Analyze и не разрешает
    персональную оценку менеджера.
15. Принятая модельная ориентация показывает `Менеджер/Клиент` и сохраняет
    источник решения; непринятая показывает `Спикер A/B`.
16. Ненулевой код завершения модели не принимается даже при наличии JSON;
    штатный `ordinary_two_party=false` без confidence кэшируется как отказ.
17. Модельное решение привязано к непустому `source_recording_id` и хешу
    текущего диалога; перенос решения на другой звонок не открывает роли.
18. При доказанной Mango-инверсии selective Resolve переставляет не только
    реплики, но и полные варианты Whisper/GigaAM по физическим дорожкам.
19. Невалидная независимая разметка отзывает подписи ролей и в текстовом
    экспорте, а не только внутри Analyze.
20. Кэшированный ответ проходит ту же нормализацию, что новый ответ; внутренние
    метаданные кэша не могут повысить confidence или подменить hash запроса.
21. После фильтрации в доверенном диалоге обязаны остаться обе физические
    дорожки; односторонний результат не открывает оценку менеджера.

## Бритва и бюджет

- Вся задача, включая исправление «вопрос + да»: не более 150 добавленных строк
  нетестового кода.
- Новых зависимостей и второго анализатора нет.
- Более простой вариант «доверять порядку каналов» отвергнут: на реальных
  данных он не доказан. Встраивание в Analyze отвергнуто: оно смешивает запись
  ролей с анализом и создаёт второй путь.

## Запреты

- Не запускать ASR, реальные Resolve/Analyze, Google write и production.
- Не менять аудио, рабочую SQLite или службы M1.
- Не принимать словарную эвристику как окончательное подтверждение роли.
- Не давать модели исправлять текст или менять отдельные реплики.

## Владение

`docs/worktrees_registry.md` проверен 25.08.2026: этот worktree является
единственным локальным владельцем текущей правки. M1 provider-selective ветка —
донор/цель передачи, она не изменяется параллельно из этого worktree.

## СТОП

- Inventory вернул `decision=stop` или нашёл другого активного владельца.
- Для проверки нужен повторный ASR, live-write или изменение рабочей базы.
- Модель может изменить текст дорожки или назначать роли отдельным репликам.
- Исправление превышает 150 строк нетестового кода или требует зависимости.

## Следующий шаг

Передать M1 пакет и провести пилот 1→10→25 без повторного ASR.
