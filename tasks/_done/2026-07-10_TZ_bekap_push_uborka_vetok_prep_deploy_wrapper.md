> DONE 2026-07-10 20:56 | ветка main | codex

> TAKE 2026-07-10 19:00 | ветка main | codex

Ветка: main
Зоны: docs/, tasks/, audits/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/
Семантический-аудит: да

# Wrapper: backup, push, deploy preparation and branch cleanup

Исполнять по внешнему ТЗ:

`/Users/dmitrijfabarisov/Claude Projects/Foton/2026-07-10_TZ_bekap_push_uborka_vetok_prep_deploy.md`

Уточнения владельца из промта:

- push консолидированного `main` в `origin` и `yandex` разрешён;
- живой бот в будущем окне останавливается полностью;
- звонковая память включается сразу после безопасного SWAP;
- сам live-stop/deploy остаётся отдельным гейтом «готов к стопу -> жми» из ТЗ;
- уборка веток выполняется только после подтверждённого подъёма нового live.

Инварианты: не писать в AMO/Tallanto/CRM, не отправлять клиентам, не
трогать `stable_runtime` до отдельного live-гейта, не удалять уникальные refs
до проверенного archive-tag в обоих remote.
