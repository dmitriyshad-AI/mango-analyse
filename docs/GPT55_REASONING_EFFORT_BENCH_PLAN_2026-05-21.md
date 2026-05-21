# План тестирования gpt-5.5 reasoning effort

Дата: 2026-05-21

Режим: безопасный offline-бенч. Не использовать сырые Telegram-диалоги, не отправлять сообщения клиентам, не писать в CRM/AMO/Tallanto, не трогать `stable_runtime`.

## 1. Цель

Понять, сколько токенов и времени уходит на средний ответ `gpt-5.5` при разных уровнях reasoning effort:

- `medium`;
- `high`;
- `xhigh`.

Дополнительно измерить качество ответов на репрезентативных задачах Mango Analyse / Telegram-пилота.

## 2. Важное ограничение по измерению токенов

Точные reasoning-токены можно получить только через OpenAI Responses API: поле `usage.output_tokens_details.reasoning_tokens`.

Если `OPENAI_API_KEY` отсутствует или модель недоступна через API, запасной путь через Codex CLI позволит сравнить wall-time и качество, но не даст точных hidden reasoning tokens. В таком случае:

- input/output tokens будут считаться приблизительно локальным токенизатором или byte/char-estimate;
- reasoning_tokens будут отмечены как `not_available`;
- результат нельзя использовать как точную финансовую оценку API.

## 3. Тестовые режимы

Основной режим:

- model: `gpt-5.5`;
- API: Responses API;
- reasoning: `medium`, `high`, `xhigh`;
- text verbosity: фиксированная, чтобы сравнивать reasoning, а не длину ответа;
- max output tokens: фиксированный лимит;
- store: false, если поддерживается клиентом;
- tools: disabled.

Запасной режим:

- `codex exec --model gpt-5.5 -c model_reasoning_effort="<effort>"`;
- `--ephemeral`;
- `--sandbox read-only`;
- structured JSON output;
- без `OPENAI_API_KEY`.

## 4. Набор задач

Использовать небольшой синтетический набор без персональных данных:

1. Telegram draft: обычный вопрос про цену.
2. Telegram draft: вопрос про рассрочку Фотон.
3. Telegram draft: вопрос про рассрочку УНПК.
4. Telegram draft: возврат денег без сбора ФИО/договора/телефона.
5. Telegram draft: юридическая угроза.
6. Telegram draft: промокод из интернета.
7. Brand separation: вопрос Фотон с упоминанием УНПК.
8. Brand separation: вопрос УНПК с упоминанием Фотон.
9. KB usefulness: налоговая справка, срок 10 дней.
10. Manager usefulness: составить полезный, но безопасный ответ.
11. Technical reasoning: найти риск в коротком псевдокоде Telegram-send.
12. Planning: выбрать безопасный следующий шаг для пилота.

Минимальный быстрый прогон: 6 задач x 3 efforts = 18 вызовов.

Расширенный прогон: 12 задач x 3 efforts = 36 вызовов.

## 5. Метрики

Токены:

- input_tokens;
- output_tokens;
- reasoning_tokens;
- visible_output_tokens = output_tokens - reasoning_tokens;
- total_tokens;
- cached_tokens, если есть;
- среднее, медиана, min, max по effort.

Время:

- wall_time_seconds;
- среднее и медиана по effort;
- ошибки и incomplete status.

Качество:

- valid_json: ответ является валидным JSON;
- route_correct: правильный маршрут `draft_for_manager` или `manager_only`;
- forbidden_terms_absent: нет запрещенных слов/обещаний;
- required_terms_present: есть обязательные элементы;
- brand_safe: не смешаны Фотон/УНПК;
- useful_for_manager: черновик не пустой и не сводится только к "уточним";
- total_quality_score от 0 до 100.

Оценка качества сначала автоматическая по рубрике. Смысловой вывод фиксировать как `semantic_signal`, а не как окончательный `semantic_pass`.

## 6. Артефакты

Создать отдельную папку:

`audits/_inbox/gpt55_reasoning_effort_bench_20260521/`

В нее сохранить:

- `cases.jsonl` - синтетические тестовые задачи;
- `raw_results.jsonl` - ответы модели и usage;
- `scored_results.csv` - плоская таблица с метриками;
- `summary.json` - агрегированные числа;
- `summary.md` - краткий бизнес-вывод;
- `run_log.txt` - команды и ошибки без секретов.

## 7. Критерий полезного результата

Результат считается полезным, если для каждого effort есть:

- минимум 6 успешных ответов;
- средний total_tokens;
- средний reasoning_tokens или явная отметка, почему точная метрика недоступна;
- средняя задержка;
- quality_score;
- список типовых отличий medium/high/xhigh.

## 8. Ожидаемый итог

На выходе должен быть практический вывод:

- какой effort брать по умолчанию для Telegram-черновиков;
- когда оправдан `high`;
- когда оправдан `xhigh`;
- насколько токены/время растут относительно `medium`;
- какие задачи реально выигрывают от более глубокого reasoning.
