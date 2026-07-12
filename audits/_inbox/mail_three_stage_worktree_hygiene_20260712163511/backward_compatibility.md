# Обратная совместимость

- Форматы: старый `max_messages=0` без явного unlimited-флага по-прежнему
  означает ноль сообщений; новый режим включается только controlled wrapper.
- Потребители: старые mail-capture/nightly задачи не переподключены и не
  перезапущены. Три новых имени задач аддитивны.
- Existing builder input `stage2_delta_full_events.jsonl` сохранён намеренно;
  новый код только добавляет явные archive DB paths.
