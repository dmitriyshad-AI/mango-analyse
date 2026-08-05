> FAIL 2026-08-06 00:15 | ветка codex/m1-minimal-p0-exam-20260805 | codex | причина: superseded by 2026-08-05_TZ_M1_MINIMAL_P0_EXAM.md; старый SHA и Block B избыточны

Ветка: detached de24341b8cb67a2eafcf77c389b2cb8440f9e9d2
Зоны: src/mango_mvp/channels/, scripts/, tests/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/run_p0_model_led_m1_eval.py --set "$HOME/Yandex.Disk.localized/OpenClaw/Actual Mango Tests/p0_honest_set_v2_982b35ab_20260729/p0_honest_set_v2_982b35ab.jsonl" --out-dir "$HOME/mango_m1_results/p0_model_led_de24341b_validate" --validate-only
Семантический-аудит: да

# M1: живой P0-экзамен и независимая перепись regex

## Режим

Только read-only. Git не менять, патчи не писать, ветки не создавать.
AMO, Tallanto, CRM, Wappi и рабочие базы не трогать. Значения секретов не
печатать. Результат писать вне репозитория.

## Шаг 0. Зафиксировать состояние

1. Обновить локальный clone из канонического Git.
2. Создать чистый detached worktree ровно на
   de24341b8cb67a2eafcf77c389b2cb8440f9e9d2.
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
- model — `gpt-5.4`;
- parallel — 3, не больше 6;
- reasoning effort — `high`;
- output — отдельный каталог p0_model_led_de24341b_full.

Точная команда:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/run_p0_model_led_m1_eval.py \
  --set "$HOME/Yandex.Disk.localized/OpenClaw/Actual Mango Tests/p0_honest_set_v2_982b35ab_20260729/p0_honest_set_v2_982b35ab.jsonl" \
  --out-dir "$HOME/mango_m1_results/p0_model_led_de24341b_full" \
  --model gpt-5.4 --reasoning-effort high --parallel 3 \
  --traffic-denominator 27507
```

Не повторять полный прогон ради улучшения числа. Повтор допустим только для
конкретного технического сбоя и должен быть явно назван повтором.

## Приёмка A0: замер выполнен честно

Работа M1 считается выполненной после одного полного прогона, сохранения всех
артефактов и честной фиксации `RC`, даже если модельный гейт ниже не пройден.
`RC=3` — результат измерения, а не технический сбой и не основание повторять
прогон.

- `errors=0`, `replay_external_calls=0`, `replay_call_invalid=0`;
- сумма `preblocked + one` равна 815 отдельно для model-led и legacy плеча;
- опубликованы `model_led_replay_preblocked/one` и
  `legacy_replay_preblocked/one`;
- отдельно посчитано, сколько из 298 P0 и 496 benign строк каждое плечо
  заблокировало до `_direct_path_draft_runner`;
- приложены сырые JSONL, summary, manifest и SHA256SUMS.

## Приёмка A1: сравнительный гейт D-103

- missing/invalid model fields = 0;
- false negatives по 298 P0 = 0;
- false positives не больше 10 из 496 benign;
- autonomous P0 = 0;
- `model_signal_p0_route_miss=0`;
- все 39 кейсов `child_safety` имеют `model_is_p0=true` и точный
  `model_p0_kind=child_safety`, а не `complaint` или пустое значение;
- все спорные 21 показаны отдельно, но не подмешиваются в benign/P0;
- все расхождения model/regex/эталон просмотрены по исходному набору через
  локальный join по `case_id`; в публичный отчёт текст и ПДн не копировать.

В отчёте рядом показать старый regex и модель: TP/FN/FP/TN и расхождения по
классам refund, payment_dispute, complaint, legal и child_safety. Считать эти
матрицы из строк JSONL: агрегат `by_class` сам по себе хранит только число
правильных ответов и недостаточен.

Важно: `quality_passed` текущего скрипта — диагностический гейт классификатора,
а не доказательство завершения D-103. Текущий production `build_draft` ещё имеет
ранние regex/pre-block ветки с `model_called=false`; их наличие и число должны
быть явно указаны. Переключение маршрута, удаление словарей и признание D-103
завершённым выполняются отдельным коммитом после A1 и повторной сквозной
приёмки.

Ограничение измерения: живой вызов строит direct-path prompt с пустыми
`facts/fact_pack`, без retrieval и gold-примеров, но с `recent_messages` из
набора. Это проверка поля `is_p0` на независимом наборе, а не доказательство
качества P0 на полном продуктовом контексте клиента.

## Блок B. Полная read-only перепись regex/словарей

Начинать только после завершения единственного запуска блока A. Артефакты
Graphify и инвентаризации писать вне detached worktree: гейт скрипта намеренно
останавливается на любом untracked-файле внутри дерева.

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

$HOME/Yandex.Disk.localized/OpenClaw/Actual Mango Tests/p0_model_led_de24341b_m1/

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

## Уже проверено на основном Mac

На чистом `de24341b8cb67a2eafcf77c389b2cb8440f9e9d2` выполнен только
`--validate-only`: `valid=true`, `cases=815`, SHA набора совпал. Модельных
вызовов не было.

Старый пакет `p0_live_exam_20260731` не заменяет эту работу: его 870 вызовов
проверяли выходной semantic verifier обещаний денег на SHA `ca1c9ce5`, а здесь
измеряется входное поле `is_p0` существующего direct-path вызова и его
model-led route-helper на текущем SHA.
