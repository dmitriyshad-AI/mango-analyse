# Tallanto API Notes (2026-04-13)

Источник:
- локальная коллекция Postman: `tallanto_postman_collection.json`
- публичный introspection endpoint: `https://kmipt.tallanto.com/service/api/rest.php`

## Базовая схема

Base URL:
- `https://kmipt.tallanto.com/service/api/rest.php`

Аутентификация:
- заголовок `X-Auth-Token: <token>`

Content-Type:
- `application/x-www-form-urlencoded`

## Подтвержденные методы API

Публичный introspection endpoint без токена возвращает сигнатуры методов:
- `list_possible_modules`
- `list_possible_fields`
- `list_possible_fields_doc`
- `list_enum_values`
- `get_entry_by_id`
- `get_entry_by_fields`
- `get_entry_list`
- `set_entry`
- `delete_entry`

## Подтвержденные шаблоны запросов из Postman-коллекции

### 1. Список модулей
GET
`/service/api/rest.php?method=list_possible_modules`

### 2. Список полей модуля
GET
`/service/api/rest.php?method=list_possible_fields&module=Contact`

### 3. HTML-документация по полям модуля
GET
`/service/api/rest.php?method=list_possible_fields_doc&module=Contact`

### 4. Список enum-значений
GET
`/service/api/rest.php?method=list_enum_values&options[]=...`

### 5. Получить запись по id
GET
`/service/api/rest.php?method=get_entry_by_id&module=Contact&id=<uuid>`

### 6. Получить запись по значению поля
GET
`/service/api/rest.php?method=get_entry_by_fields&module=Contact&fields_values[phone_mobile]=7978...`

### 7. Получить список записей
POST
`/service/api/rest.php?method=get_entry_list&module=Contact&select_fields[]=first_name&select_fields[]=last_name`

Body (`x-www-form-urlencoded`) пример:
- `query=DATE(contacts.date_entered) BETWEEN "2024-01-01" AND "2025-06-01"`
- `offset=0`
- `order_by=last_name ASC`

### 8. Создать запись
POST
`/service/api/rest.php?method=set_entry&module=Contact`

Body пример:
- `fields_values[first_name]=Игорь`
- `fields_values[last_name]=Васильков`
- `fields_values[email1]=...`
- `fields_values[phone_mobile]=7978...`
- `fields_values[assigned_user_name]=...`
- `fields_values[type_client_c]=Потенциальный клиент`
- `fields_values[filial]=...`

### 9. Обновить запись
POST
`/service/api/rest.php?method=set_entry&module=Contact&id=<uuid>`

Body:
- `fields_values[field_name]=value`

### 10. Удалить запись
GET
`/service/api/rest.php?method=delete_entry&module=Contact&id=<uuid>`

## Что уже ясно для будущей интеграции Mango analyse

Основной путь матчинга можно строить так:
- `phone -> Contact`
- затем читать связанные поля контакта / учебный контекст
- при необходимости читать дополнительные модули по id/field match

Полезные подтвержденные возможности API:
- прямой поиск по значению поля (`get_entry_by_fields`)
- выборка списком с query (`get_entry_list`)
- introspection модулей и полей
- чтение enum-значений справочников

## Ограничение, выявленное в live-проверке

Переданный токен в live-вызове сейчас дает `401 Unauthorized`.
Это означает одно из двух:
- токен устарел / неверен
- токен ограничен по среде или источнику запроса

Поэтому на 2026-04-13 подтверждены:
- схема API
- схема аутентификации
- структура методов
- шаблоны запросов

Но live-чтение бизнес-данных из Tallanto еще требует отдельной валидации рабочего токена.

## Практический следующий шаг

Когда будет подтвержден рабочий токен, первым делом стоит реализовать в Mango analyse:
1. `list_possible_modules`
2. `list_possible_fields` для целевых модулей
3. `get_entry_by_fields` для поиска контакта по телефону
4. `get_entry_list` для выборки учебного контекста по найденному контакту

