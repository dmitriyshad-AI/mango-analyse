## Финальный аудит проекта на 22.03.2026

### Что улучшено в этом цикле

1. Уточнена классификация звонков в `Analyze`
   - уменьшена доля ложных `sales_call`
   - действующие клиенты и обратная связь чаще уходят в `existing_client_progress` / `service_call`
   - технические кейсы устойчивее попадают в `technical_call`

2. Добавлен слой ручного контроля качества
   - `quality_flags.needs_review`
   - `quality_flags.review_reasons`
   - дублирующие top-level поля `needs_review`, `review_reasons`

3. Улучшена нормализация `next_step`
   - английские формулировки нормализуются в русский
   - channel-heavy формулировки вроде Telegram / email / WhatsApp сводятся к стабильному действию
   - при этом русская конкретика вроде `Перезвонить завтра` сохраняется

4. Улучшен Excel-экспорт
   - в `Calls` добавлены:
     - `call_type`
     - `needs_review`
     - `review_reasons`
   - в `Contacts` добавлены:
     - `latest_call_type`
     - `needs_review`
     - `review_reasons_latest`

### Эффект на первой 1000 analyzed calls

До финального цикла:

- `sales_call = 609`
- `service_call = 149`
- `technical_call = 67`
- `existing_client_progress = 16`
- `non_conversation = 159`
- `sales_without_product = 274`
- `sales_without_next_step = 143`
- `sales_without_product_and_next_step = 107`

После финального цикла:

- `sales_call = 555`
- `service_call = 175`
- `technical_call = 70`
- `existing_client_progress = 41`
- `non_conversation = 159`
- `needs_review = 261`
- `sales_without_product = 220`
- `sales_without_next_step = 110`
- `sales_without_product_and_next_step = 74`

### Что это означает

- Продажных звонков стало меньше, но классификация стала честнее.
- Существенная часть действующих клиентов и сервисных кейсов перестала маскироваться под продажи.
- Вместо “тихих ошибок” теперь есть явная очередь сомнительных звонков через `needs_review`.

### Что еще остается как техдолг

1. `sales_service_overlap`
   - сейчас это самый важный remaining bucket
   - в первой тысяче таких кейсов `188`
   - это правильная очередь для селективного rerun / ручной проверки

2. `long_non_conversation`
   - осталось `3` кейса
   - это маленький, но качественно важный хвост

3. Старые смысловые ошибки LLM
   - детерминированная миграция не может перепридумать summary заново
   - для действительно ошибочного старого `history_summary` нужен селективный re-analyze

### Вывод

Проект находится в сильном предбоевом состоянии:

- ASR-часть рабочая
- `Analyze` уже пригоден для массовой витрины
- Excel теперь несет не только данные, но и QC-сигналы

Следующий правильный шаг:

1. не запускать сразу еще тысячи звонков вслепую;
2. сначала выбрать `needs_review` из первой 1000;
3. проверить 30-50 строк руками;
4. затем масштабировать анализ дальше.
