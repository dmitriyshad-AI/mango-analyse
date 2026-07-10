# Риски

- Клиентский риск: профильный prompt теперь добавляет inline SemanticFrame и active semantic reading classes. Поведение меняется только на этой ветке; живой бот не трогался. Smoke подтвердил frame/trace emission, но это не semantic_pass.
- Данные/записи: внешних write-операций нет; M1, Telegram polling, AMO, Tallanto, CRM не запускались.
- Откат: revert коммита Блока 1 возвращает `SEMANTIC_FRAME_SHADOW` к default-off и убирает CSV-дефолт reading classes.
