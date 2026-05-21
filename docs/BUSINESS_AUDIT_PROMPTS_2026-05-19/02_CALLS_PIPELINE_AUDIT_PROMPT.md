# Промт 2: аудит звонков и обработки разговоров

```text
Ты работаешь в проекте:
/Users/dmitrijfabarisov/Projects/Mango analyse

Задача: провести read-only бизнес-аудит слоя звонков и обработки разговоров.

Режим: ничего не менять, не запускать ASR, не запускать Resolve+Analyze по реальным данным, не менять stable_runtime. Только чтение и анализ.

Что проверить:
1. Как сейчас устроен путь звонка:
   - скачивание из Mango;
   - хранение аудио;
   - ASR;
   - Resolve;
   - Analyze;
   - canonical master;
   - phone-chain;
   - AMO/contact export;
   - связь с Customer Timeline и deal-aware.
2. Что уже улучшено в analyze X-v3, но ещё не переобработано на исторических данных.
3. Нужно ли делать selective re-analyze 200-500 звонков и по каким критериям.
4. Нужно ли делать полный re-analyze исторических звонков или это избыточно.
5. Какие артефакты звонков актуальные, какие старые, какие нельзя трогать.
6. Достаточно ли текущей истории звонков для Telegram-бота и CRM-сводок.
7. Где есть риск плохих данных:
   - несодержательные звонки;
   - неверная склейка спикеров;
   - старые LLM-сводки;
   - обрезанные истории;
   - устаревший next step.
8. Какие тесты покрывают качество звонков, а какие только форму.

Обязательные файлы для чтения:
- AGENTS.md
- docs/CURRENT_STATE.md
- docs/AUDIO_STORE_RUNTIME_CONTRACT_2026-05-16.md
- docs/CANONICAL_MASTER_BUILD_REPORT_2026-05-09.md
- docs/RESOLVE_LLM_DISABLED_2026-05-15.md
- docs/CUSTOMER_TIMELINE_INTEGRATION_DECISION_2026-05-15.md
- docs/DEAL_AWARE_STAGE709_REVIEW_2026-05-14.md
- stable_runtime/CURRENT_RUNTIME.json только read-only
- последние audit packs по runtime rebuild и звонкам

Дополнительно посмотреть:
- scripts, связанные с mango/canonical/asr/resolve/analyze/export;
- tests по analyze, resolve, canonical master, transcript quality.

Особое внимание:
- не путай “все звонки обработаны” с “данные достаточно качественные для бота”;
- отдельно выдели, какие данные нужны именно для ответа клиенту, а какие только для аналитики;
- найди неиспользуемые, но полезные модули по истории звонков.

Итоговый файл:
docs/BUSINESS_MODULE_AUDIT_CALLS_PIPELINE_2026-05-19.md

В конце дай:
1. Можно ли считать звонковый слой достаточным для пилота.
2. Какие 3-5 улучшений дадут максимальную пользу.
3. Нужно ли переобрабатывать данные сейчас.
4. Что строго нельзя трогать.
```

