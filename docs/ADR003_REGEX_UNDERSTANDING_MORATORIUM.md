# ADR-003 Regex Understanding Moratorium

Дата: 2026-07-01

## Правило

Смысл клиентского сообщения в direct-path должен определять SemanticFrame/LLM, а не новый regex или keyword-детектор.

Новый клиентский смысловой сбой проходит через такой путь:

1. добавить пример в eval-набор;
2. улучшить инструкцию/схему/калибровку SemanticFrame;
3. проверить метриками и сырьем;
4. только потом включать поведение за флагом.

Нельзя добавлять новый `re.compile` или keyword-таблицу, которая читает сырой текст клиента и решает:

- P0/не P0;
- intent/topic;
- close/not-close;
- requested action;
- answerability/relevance;
- venue/scope/product meaning;
- готовность к оплате или сделке.

## Что остается разрешенным детерминизмом

Разрешены механические проверки выхода и инфраструктурные парсеры:

- ПДн/телефон/email/id scrub;
- проверка чисел, дат, ссылок, брендов и обещаний в тексте ответа;
- fail-closed, когда модель не ответила или дала низкую уверенность;
- тестовые/отчетные парсеры.

Любое расширение regex в runtime-канале должно явно объяснить, что это проверка выхода, а не понимание клиента, и обновить guard-тест.

## CI guard

Мораторий закреплен тестом `tests/test_adr003_regex_understanding_moratorium.py`.

Он проверяет два frozen snapshot:

- `tests/fixtures/adr003_runtime_channel_regex_snapshot.json` — все текущие `re.compile` в runtime-каналах;
- `tests/fixtures/adr003_direct_path_text_patterns_snapshot.json` — direct-path `re.compile`, inline `re.search/sub/...`, верхнеуровневые keyword/marker таблицы и строковые `"..." in text`-проверки в файлах понимания.

Если тест упал, нельзя просто обновить snapshot. Сначала надо ответить:

1. это проверка выхода/fail-closed/PII/brand/fabrication, а не понимание сырого клиентского текста?
2. есть ли eval-кейс, который фиксирует найденный смысловой сбой?
3. почему это не должно решаться SemanticFrame?

Если это новый смысл клиента, snapshot не обновляется: добавляется eval-кейс и калибруется SemanticFrame. Если это разрешенная механическая проверка выхода, в audit pack нужно явно написать причину и только затем обновить snapshot.

## Разрешенное обновление 2026-07-01: SemanticFrame manager-action gate

Snapshot `tests/fixtures/adr003_direct_path_text_patterns_snapshot.json` обновлен из-за новых технических констант:

- `TELEGRAM_SEMANTIC_FRAME_MANAGER_ACTION_GATE`;
- schema/version и порог confidence;
- закрытые enum-наборы `deal_stage`/`payment_readiness`, которые читают уже готовый `semantic_frame`.

Это не новое regex/keyword-понимание сырого клиентского текста. Гейт не парсит сообщение клиента и не добавляет словари маркеров. Он работает только по posthoc `semantic_frame` со статусом `ok`, за default-OFF флагом, и может только повысить автономный маршрут до `draft_for_manager`; текст ответа не меняет.
