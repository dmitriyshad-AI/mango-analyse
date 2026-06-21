# Backward compatibility

## OFF-поведение

Если `TELEGRAM_BOT_SAFE_CRM_CONTEXT` выключен, bot-safe блок не добавляется, поведение не меняется.

## Данные

SQLite, read_api и runtime builder не менялись.

## Prompt

Изменено только представление bot-safe памяти внутри direct-path prompt. Блок «Факты по вашему вопросу» не фильтруется.
