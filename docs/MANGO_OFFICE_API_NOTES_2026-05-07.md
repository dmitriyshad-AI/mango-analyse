# Mango Office API notes for shadow polling POC

Дата: 2026-05-07

Цель: зафиксировать минимальный read-only POC для получения истории звонков Mango Office без запуска ASR/R+A, без записи в runtime-БД и без записи в AMO/Tallanto.

## Официальные источники

- `https://www.mango-office.ru/support/api/`
- `https://www.mango-office.ru/support/integratsiya-api/restapivatshelp/`
- `https://www.mango-office.ru/support/integratsiya-api/restapivatshelp/obshchie_voprosy_po_api_vats_mango_office/`
- `https://www.mango-office.ru/support/integratsiya-api/chasto_zadavaemye_voprosy/obshchieyvoprosy/otsutstvuyutyzapisiyrazgovorov/`

## Что подтверждено документацией

Mango Office публикует несколько API-направлений. Для этой ветки нужен API Виртуальной АТС: он относится к управлению облачной телефонией и поддерживает получение истории звонков и получение записей разговоров.

Официальный пример истории звонков использует двухшаговый flow:

1. `POST https://app.mango-office.ru/vpbx/stats/request`
2. `POST https://app.mango-office.ru/vpbx/stats/result`

Запрос подписывается так:

```text
sign = sha256(vpbx_api_key + json + api_salt)
```

POST body отправляется как form fields:

```text
vpbx_api_key=<api key>
sign=<sha256 hex>
json=<json payload>
```

В официальном примере для истории передаются:

```json
{
  "date_from": "1481630491",
  "date_to": "1481734491",
  "from": {
    "extension": "",
    "number": ""
  },
  "to": {
    "extension": "",
    "number": ""
  },
  "fields": "records,start,finish,answer,from_extension,from_number,to_extension,to_number,disconnect_reason,line_number,location,entry_id"
}
```

## Записи разговоров

Для доступности записей должны быть выполнены условия:

- в ВАТС подключена запись разговоров;
- в настройках API включена возможность генерации и использования ссылок на записи;
- конкретный звонок должен иметь сохраненную запись.

В документации и FAQ отдельно отмечается, что отсутствие записей часто связано именно с настройками записи/ссылок. Также для коротких разговоров запись может не сохраниться.

## Фактическая проверка 2026-05-07

Реальный `stats/request` с credentials вернул JSON:

```json
{"key":"..."}
```

Реальный `stats/result` вернул не JSON, а `text/csv;charset=UTF-8` без header row. Порядок полей соответствует запрошенной строке `fields`:

```text
records;start;finish;answer;from_extension;from_number;to_extension;to_number;disconnect_reason;line_number;location;entry_id
```

Пример формы одной строки без персональных данных:

```text
[];1778150907;1778150929;0;202;sip:user@example;;79000000000;1121;74950000000;abonent;...
```

Вывод для кода: `stats/result` нужно парсить как CSV с `;`, а `records=[]` считать отсутствием записи. Непустой `records` приходит в квадратных скобках и может быть использован как `recording_id` для отдельного read-only шага получения ссылки на запись.

Первый shadow poll на последние 2 часа:

```text
source_rows = 69
normalized_events = 69
normalization_errors = 0
enqueue_shadow_capture = 38
skip_no_recording = 30
skip_duplicate = 1
```

## Неизвестное после первой проверки

Эти детали нельзя считать закрытыми до ручного запроса с реальными credentials:

- задержка появления записи после завершения звонка;
- rate limits для нашего тарифа;
- максимальный размер окна `date_from/date_to`;
- timezone в полях `start`, `finish`, `answer` фактически выглядит как Unix timestamp, нужно подтвердить на ручной сверке с кабинетом;
- какие поля нужны для внутренних/переведенных звонков.

## Shadow polling POC

Первый POC должен делать только это:

1. Взять окно времени, например последние `2` часа.
2. Выполнить `stats/request`.
3. Выполнить `stats/result`.
4. Нормализовать каждую строку в `TelephonyCallEvent`.
5. Пропустить через `CapturePlanner`.
6. Напечатать или сохранить JSON report:
   - `enqueue_shadow_capture`
   - `skip_duplicate`
   - `skip_no_recording`
   - normalization errors

Запрещено в этом POC:

- скачивать аудио;
- писать в `stable_runtime`;
- писать в текущую SQLite runtime-БД;
- запускать ASR/R+A;
- писать в AMO/Tallanto;
- менять batch/start/run-ui scripts.

## Environment variables

```text
MANGO_OFFICE_API_KEY=<vpbx api key>
MANGO_OFFICE_API_SALT=<vpbx api salt/signature secret>
MANGO_OFFICE_BASE_URL=https://app.mango-office.ru
```

## Dry-run command

```zsh
PYTHONPATH=src python3 scripts/mango_office_shadow_poll.py \
  --tenant foton \
  --hours 2 \
  --out /tmp/mango_shadow_poll_report.json
```

The script is intentionally read-only. Existing seen keys may be supplied from a small text file:

```zsh
PYTHONPATH=src python3 scripts/mango_office_shadow_poll.py \
  --tenant foton \
  --hours 2 \
  --seen-keys /tmp/mango_seen_keys.txt
```

This file is only an input for dedupe simulation; the script does not mutate it.

The script loads `.env` automatically when `python-dotenv` is installed.
