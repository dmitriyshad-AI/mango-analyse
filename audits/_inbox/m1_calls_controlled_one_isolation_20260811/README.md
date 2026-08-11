# M1 Calls controlled-one isolation audit

Пакет фиксирует приёмку коммита
`0027677eba2539968ce6e8183ed5413a9f3de07a` в ветке
`codex/m1-calls-service-fast-value-20260811`.

Результат: кодовая и синтетическая готовность controlled-one — GO. Реальный
пилот, служба, cutover и бизнес-польза — STOP до отдельного разрешения Дмитрия,
переноса доказанного состояния и человеческой проверки результата.

Главные гарантии блока:

- один owner-authorized `source_call_id`, связанный с точной строкой и аудио;
- порядок Whisper MLX -> очистка кэша -> GigaAM -> Resolve -> Analyze;
- controlled-подготовка не создаёт общий service lineage marker;
- ticket использует SHA именно проверенного cutover manifest;
- остальные строки БД и исходное аудио контролируются до/после;
- ошибки уборки приватной аудиокопии не уничтожают доказательство стадий и не
  превращаются в успешный пилот;
- AMO, публикация, capture и широкие service-команды в controlled scope закрыты.

Флаги приёмки:

- `formal_pass=true`;
- `semantic_pass=true`;
- `data_pass=synthetic_only`;
- `business_pass=false`;
- `runtime_pass=false`.

Пакет не содержит текста звонков, телефонов, email, токенов или секретов.
