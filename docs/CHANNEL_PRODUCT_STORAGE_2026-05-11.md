# Channel product storage

Дата: 2026-05-11

## Зачем это нужно

До этого channel/bot слой уже умел принимать входящее сообщение, строить
безопасный черновик ответа и предлагать действия для менеджера. Главный недочет:
часть состояния жила в памяти процесса. После перезапуска такой контур нельзя
показывать как продуктовый рабочий стол.

Этот шаг добавляет локальное постоянное хранилище для клиентского ноутбука или
сервера. Оно не заменяет основной pipeline обработки звонков и не пишет в
runtime DB.

## Что появилось

Добавлен `ChannelSQLiteStore` в `src/mango_mvp/channels/persistence.py`.

Он хранит:

- сессии переписки;
- входящие сообщения без `raw_payload`;
- черновики ответов;
- рекомендуемые действия;
- историю review-событий;
- решения signal engine;
- feedback/outcome-события, импортированные только как read-only факты.

Добавлен read-only workspace слой в `src/mango_mvp/channels/workspace.py`.

Он собирает операторский экран:

- какие сессии требуют проверки;
- сколько черновиков ждут менеджера;
- сколько действий предложено;
- где есть hot lead;
- где есть рискованные feedback-события;
- общий safety contract.

Добавлен demo-сценарий:

- `src/mango_mvp/channels/demo.py`;
- `scripts/build_channel_workspace_demo.py`.

Он создает три безопасные демонстрационные переписки: Telegram, чат сайта и CRM
чат. Все сообщения искусственные, без реальных credentials и без live send.

## Границы безопасности

Новый слой:

- не вызывает сеть;
- не отправляет сообщения в Telegram/Web/CRM chat;
- не пишет в AMO, Tallanto или CRM;
- не запускает ASR;
- не запускает Resolve+Analyze;
- не пишет в `stable_runtime`;
- не сохраняет `raw_payload` по умолчанию;
- запрещает live-статусы `sent`, `executed`, `live_sent`.

SQLite-файл предназначен только для product/channel состояния, например:

```text
product_root/channel/channel_product.sqlite
```

Путь внутри `stable_runtime` и runtime-looking имена вроде `runtime.db`
отклоняются.

## Как протестировать

Focused тесты:

```bash
cd "/Users/dmitrijfabarisov/Projects/Mango analyse"
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q \
  tests/test_channels_demo.py \
  tests/test_channels_persistence.py \
  tests/test_channels_workspace.py \
  tests/test_channels_storage.py \
  tests/test_channels_feedback.py \
  tests/test_channels_signals.py \
  tests/test_channels_preview_service.py \
  tests/test_channels_actions.py
```

Ожидаемый результат:

```text
52 passed
```

Полный channel smoke:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_channels_*.py
```

Ожидаемый результат:

```text
91 passed
```

Локальная demo-команда:

```bash
cd "/Users/dmitrijfabarisov/Projects/Mango analyse"
PYTHONPATH=src python3 scripts/build_channel_workspace_demo.py \
  --db-path /private/tmp/mango_channel_demo.sqlite \
  --pretty
```

## Что это дает для SaaS уровня

Это первый шаг от "бот отвечает в памяти" к продуктовой модели:

1. Сообщение клиента сохраняется как нормальный объект продукта.
2. Черновик ответа не теряется после перезапуска.
3. РОП/оператор может видеть очередь проверки.
4. Сигналы и feedback можно использовать для будущего обучения поведения.
5. Контур остается безопасным: пока это только preview/read-only, без live send.

## Что делать дальше

Следующий безопасный шаг:

1. Подключить workspace summary к read-only Product API после стабилизации
   параллельных изменений в `product_api.py`.
2. Добавить UI approval workspace: панели "Нужна проверка", "Горячие лиды",
   "Риски", "Feedback".
3. Подготовить controlled-send design, но не включать live send без отдельного
   approval gate.
4. Когда обработка звонков стабилизируется, связать знания из звонков с channel
   draft context.
