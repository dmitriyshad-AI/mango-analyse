# План быстрого восстановления AMO OAuth и live shadow-run

## Текущее состояние на 14 апреля 2026

Факт по текущей amoCRM интеграции:
- access token из `amo_integration_connections` дает `401 Unauthorized`
- refresh flow дает `401 Token has been revoked`
- значит текущая OAuth row в shared DB больше не пригодна для live-чтения AMO

Это не проблема очереди или LLM. Это проблема живой авторизации.

## Что уже исправлено в коде Mango analyse

В runtime ужесточены два слабых места:

1. `refresh_connection_tokens(...)` теперь делает refresh под `SELECT ... FOR UPDATE`
- это уменьшает гонки, когда несколько процессов одновременно пытаются refresh-ить один и тот же OAuth row

2. `/status` больше не должен врать, что интеграция "connected", если токен уже протух или connection ушла в `reauthorization_required`
- при stale token статус должен быть `token_stale`
- при revoked refresh token статус должен стать `reauthorization_required`

Файлы:
- `src/mango_mvp/amocrm_runtime/amo_integration.py`
- `tests/test_amocrm_deals.py`

## Что нужно сделать прямо сейчас

### Шаг 1. Переавторизовать amoCRM

Нужна именно полная reauthorization существующей внешней интеграции, а не попытка оживить старый refresh token.

Безопасный путь:
- открыть существующий UI/кнопку подключения amoCRM для интеграции `AI Office`
- пройти OAuth заново до успешного callback
- убедиться, что в shared DB row обновились `access_token`, `refresh_token`, `expires_at`

Важно:
- использовать ту же интеграцию, которая уже связана с `https://api.fotonai.online/api/integrations/amocrm/callback`
- не создавать второй параллельный контур авторизации для того же runtime без необходимости

### Шаг 2. Перезапустить локальный runtime Mango analyse

Нужно, чтобы локальный runtime подхватил обновленный код и свежую OAuth row.

Рекомендуемый запуск:
```bash
cd "/Users/dmitrijfabarisov/Projects/Mango analyse"
./stable_runtime/run-amocrm-runtime.sh
```

Если runtime уже висит на порту, сначала остановить старый процесс и поднять новый.

### Шаг 3. Проверить статус

Проверка:
```bash
curl -sS -H 'X-API-Key: ai-office-local-key' http://127.0.0.1:8010/api/integrations/amocrm/status
```

Нормальный результат:
- `connected: true`
- `token_source: oauth`
- `status: active`
- `access_token_present: true`
- `refresh_token_present: true`
- `last_error: null`

Дополнительная проверка:
```bash
curl -sS -X POST -H 'X-API-Key: ai-office-local-key' http://127.0.0.1:8010/api/integrations/amocrm/refresh
```

Если refresh проходит, контур снова живой.

### Шаг 4. Только после этого запускать live 30-day shadow-run

Через runtime, не прямыми ad-hoc python-скриптами к shared DB.

## Почему OAuth упал

По фактам, которые подтверждены здесь:
- текущий refresh token revoked
- предыдущие параллельные direct-runner процессы создавали contention вокруг `amo_integration_connections`
- status endpoint раньше считал наличие токена за `connected`, даже если токен уже непригоден

Точный внешний триггер revoke мы здесь не докажем без журналов amoCRM, но наиболее вероятные причины такие:
- повторная авторизация интеграции в другом контуре
- замена/перепривязка клиентских секретов интеграции
- ручной revoke или сброс токенов на стороне amoCRM
- гонки refresh при нескольких конкурирующих процессах

## Что делать, чтобы живой AMO OAuth не падал

### 1. Один владелец refresh-контура
Не запускать несколько независимых клиентов, которые сами читают shared DB row и refresh-ят токен.

Правильная схема:
- один долгоживущий runtime
- все чтение AMO через него
- никакие отдельные `python`-скрипты не должны параллельно дергать refresh напрямую

### 2. Не считать shared DB row публичной точкой доступа
Рабочий доступ к AMO должен идти через runtime API, а не через прямые разовые SQL+HTTP скрипты.

Практически:
- `curl -> runtime endpoint`
- а не `python -> resolve_amo_access_context()` в нескольких отдельных процессах одновременно

### 3. Делать раннюю проверку статуса перед длинными batch-джобами
Перед любым большим shadow-run:
1. `/status`
2. `/refresh`
3. только потом запуск очереди

### 4. Не держать fallback token как основную стратегию
Direct token годится только как временный аварийный обход.
Он не должен быть главным продовым контуром.

### 5. Не плодить несколько OAuth контуров для одной и той же business-integratoin без нужды
Лучше одна понятная интеграция и один источник истины по токенам, чем несколько разрозненных попыток авторизации.

### 6. Мониторить признаки деградации заранее
Плохие признаки:
- `status = token_stale`
- `status = reauthorization_required`
- `last_error != null`
- refresh начинает отдавать `401`

Если это видим, не запускать большие очереди, а сначала чинить OAuth.

## Что делать после восстановления

### Проверка работоспособности
1. `status`
2. `refresh`
3. `contact-fields/sync` по желанию
4. короткий smoke-read одной сделки/контакта

### Затем live shadow-run
Цель:
- честно перепрогнать все закрытые сделки за последние 30 дней на живом AMO контуре
- без write-back

Режим:
- `apply_writeback = false`
- только `shadow`

## Операционный принцип

Сначала:
- восстановить OAuth
- подтвердить статус
- прогнать live 30-day shadow-run
- собрать новый ROP workbook

Только потом:
- обсуждать controlled write-back в сделки

