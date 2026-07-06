# Риски

- Клиентский риск: текущий код не меняет боевого бота и не отправляет сообщения. Replay runner пока fake-provider only.
- Данные/записи: raw export ограничен `~/.mango_local/replay_exam/raw/`; exporter test проверяет запрет произвольного пути. Live Wappi read без `--allow-live-wappi-read` невозможен.
- ПДн: pseudonymizer покрывает текст, вложенные поля, имена, телефоны, email, URL, договоры; judge payload должен получать только scrubbed data.
- Инфра-риск: реальный pilot-10 ещё не проверен на live Wappi API, потому что это требует отдельного подтверждения. Возможные API-несовпадения должны ловиться на pilot-10 до полного M1.
- Откат: пакет изолирован в новых файлах `mango_mvp.replay_exam` и scripts; revert коммита не затрагивает direct-path runtime.
