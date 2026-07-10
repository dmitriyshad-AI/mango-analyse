# Риски

- Клиентский риск: клиентам ничего не отправлялось; live bot не трогался. Риск появляется только при будущем включении памяти/CRM-write, поэтому текущий результат не считать production-ready.
- Данные/записи: prod DB, AMO, Tallanto и CRM не записывались. Staging DB менялась локально. `.codex_local` содержит ПДн и не должен попадать в git/Foton.
- AMO write risk: readiness diff показал 18 target fields would_change; 16 пустые, 2 непустые contact-поля будут затронуты. Нужен anti-clobber/pre-patch gate.
- Wappi risk: 138 `medium_text` кандидатов слабые; это только ручная подсказка, не автопривязка.
- Nightly risk: AMO incremental был page-capped (`max_pages=2`), значит не доказывает полноту боевого nightly.
- Откат: SWAP проверен на локальной копии; реальный rollback должен выполняться человеком по отдельному пакету и только после бэкапа/окна.
