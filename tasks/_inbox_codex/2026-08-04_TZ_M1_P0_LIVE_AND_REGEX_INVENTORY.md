Ветка: detached 5b444302cd8f12a1e7c14942b531b230ad24063f
Зоны: src/mango_mvp/channels/, scripts/, tests/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/run_p0_model_led_m1_eval.py --set "$HOME/Yandex.Disk.localized/OpenClaw/Actual Mango Tests/p0_honest_set_v2_982b35ab_20260729/p0_honest_set_v2_982b35ab.jsonl" --out-dir "$HOME/mango_m1_results/p0_model_led_5b444302_validate" --validate-only
Семантический-аудит: да

# M1: живой P0-экзамен и независимая перепись regex

## Режим

Только read-only. Git не менять, патчи не писать, ветки не создавать.
AMO, Tallanto, CRM, Wappi и рабочие базы не трогать. Значения секретов не
печатать. Результат писать вне репозитория.

## Шаг 0. Зафиксировать состояние

1. Обновить локальный clone из канонического Git.
2. Создать чистый detached worktree ровно на
   5b444302cd8f12a1e7c14942b531b230ad24063f.
3. Зафиксировать SHA, git status, Python и модель.
4. Проверить SHA набора:
   00067d63473cbb6000311f1828e0845c638001ee4d61935ad45308dba7c24450.
5. Подтвердить 815 строк: 298 P0, 496 benign, 21 ambiguous.

Несовпадение любого значения — STOP без полного запуска.

## Блок A. Один живой P0-прогон

Сначала выполнить validate-only из шапки. После PASS выполнить ровно один
полный запуск существующего scripts/run_p0_model_led_m1_eval.py:

- set — точный файл выше;
- traffic denominator — 27507;
- parallel — не больше 6;
- reasoning effort — высокий доступный;
- output — отдельный каталог p0_model_led_5b444302_full.

Не повторять полный прогон ради улучшения числа. Повтор допустим только для
конкретного технического сбоя и должен быть явно назван повтором.

## Приёмка A

- LLM errors = 0;
- missing/invalid model fields = 0;
- false negatives по 298 P0 = 0;
- false positives не больше 10 из 496 benign;
- autonomous P0 = 0;
- все спорные 21 показаны отдельно, но не подмешиваются в benign/P0;
- приложены сырые JSONL, summary, manifest и SHA256SUMS.

В отчёте рядом показать старый regex и модель: TP/FN/FP/TN и расхождения по
классам refund, payment_dispute, complaint/legal и child_safety.

## Блок B. Полная read-only перепись regex/словарей

Использовать свежий локальный Graphify на этом SHA только как карту. Каждый
вывод перепроверить в исходниках и реальных entrypoints.

Пересчитать все re.compile/re.search/re.match/re.findall и смысловые словари в
src и scripts. Каждая запись получает ровно один класс:

- SEMANTIC_OWNER;
- OUTPUT_FLOOR;
- PARSER_FORMAT;
- TEST_ONLY;
- DEAD_DUPLICATE.

Для SEMANTIC_OWNER дополнительно:

- какая ветка меняет маршрут или не вызывает модель;
- какое уже существующее модельное поле способно быть владельцем;
- точный production caller;
- какой тест доказывает живой путь.

Для DEAD_DUPLICATE:

- Graphify;
- rg/AST;
- scripts/CLI/launchd/runtime;
- git history;
- предлагаемый мутационный или сквозной тест.

Ничего не удалять и не предлагать новый модельный вызов.

## Приёмка B

- total = сумма пяти классов;
- unassigned = 0;
- отдельный список ранних pre-block веток, где model_called=false;
- отдельный список route/autonomy владельцев;
- отдельный список выходных проверок, которые толкуют смысл текста;
- каждое число имеет SHA, команду и путь.

## Результат

Каталог:

$HOME/Yandex.Disk.localized/OpenClaw/Actual Mango Tests/p0_model_led_5b444302_m1/

Внутри:

- REPORT.md;
- p0 full output;
- regex_inventory.csv;
- semantic_owners.md;
- dead_duplicate_candidates.md;
- commands.txt;
- SHA256SUMS.txt.

В финальном сообщении только числа, путь и вердикты A/B. Manifest писать
последним.

## Стоп

- SHA кода или набора не совпал;
- validate-only не прошёл;
- нужен Git write или изменение кода;
- доступ к модели отсутствует;
- полный прогон уже был выполнен на этом SHA.
