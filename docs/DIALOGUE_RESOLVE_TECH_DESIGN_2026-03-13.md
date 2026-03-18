# Dialogue-Level Resolve: Technical Design

Дата: 2026-03-13

## 1. Зачем меняем архитектуру

Текущий `Codex merge` работает в основном на уровне отдельных ролей:

- на стадии `transcribe` LLM сливает `variant_a` и `variant_b` отдельно для менеджера и отдельно для клиента;
- на стадии `resolve` для `stereo` LLM тоже работает по ролям, а не по последовательности реплик;
- `dialogue_lines` строятся после этого rule-based кодом.

Следствие:

- LLM почти не влияет на качество очередности реплик;
- LLM не может надежно исправлять локальные перестановки менеджер/клиент;
- финальный timed-dialogue получается не из решения LLM, а из набора эвристик.

Для качества CRM-аналитики нужен следующий уровень: LLM должна видеть весь разговор как последовательность turns и возвращать структурированный диалог.

## 2. Цель

Сделать новую стадию `dialogue-level resolve`, которая:

- получает весь звонок как список реплик с таймкодами, ролями и вариантами ASR;
- улучшает текст каждой реплики на основе `Whisper/MLX + GigaAM`;
- проверяет локальные ошибки последовательности реплик;
- возвращает новый `dialogue_lines`, пригодный для экспорта и для `Analyze`;
- не придумывает новых фактов и не превращает разговор в свободный пересказ.

## 3. Не цели

- не заменять ASR одной LLM;
- не восстанавливать слова, которых нет ни в одном ASR-варианте;
- не менять крупно весь порядок разговора;
- не генерировать summary на стадии `Resolve`.

## 4. Новый pipeline

1. `Primary ASR`
   - `mlx-whisper`
2. `Secondary ASR`
   - `gigaam`
3. `Baseline dialogue builder`
   - текущая сборка `dialogue_lines` по сегментам и rule-based постфильтрам
4. `Dialogue Resolve`
   - новый `Codex CLI` проход по всему диалогу
5. `Rescue ASR`
   - включается только если baseline/dialogue-resolve ниже порога
6. `Analyze`
   - работает только по финальному `dialogue_lines`

## 5. Принцип работы новой LLM

LLM не должна писать свободный текст.

LLM должна:

- получить фиксированный массив turns;
- для каждого turn выбрать лучший текст из `variant_a`, `variant_b`, `baseline_text`;
- при необходимости смешать тексты в пределах одной реплики;
- опционально удалить явный артефакт/эхо;
- в редких случаях поменять местами только соседние turns.

LLM не должна:

- добавлять новые turns;
- произвольно менять таймкоды;
- склеивать длинные куски в новые блоки;
- менять говорящего без достаточного основания.

## 6. Новый входной JSON для Dialogue Resolve

Новая функция должна собирать payload примерно такого вида:

```json
{
  "schema_version": "dialogue_resolve_v1",
  "call_id": 123,
  "source_filename": "20260301_123000_Иванов_...",
  "manager_name": "Иванов",
  "mode": "stereo",
  "duration_sec": 298.5,
  "providers": {
    "primary": "mlx",
    "secondary": "gigaam",
    "merge_provider": "codex_cli"
  },
  "turns": [
    {
      "turn_id": 1,
      "ts_sec": 2.1,
      "ts_label": "00:02.1",
      "speaker": "manager",
      "speaker_label": "Менеджер (Иванов)",
      "baseline_text": "Здравствуйте, меня зовут...",
      "variant_a": "Здравствуйте, меня зовут...",
      "variant_b": "Здравствуйте, меня зовут...",
      "channel_source": "left",
      "flags": []
    },
    {
      "turn_id": 2,
      "ts_sec": 3.4,
      "ts_label": "00:03.4",
      "speaker": "client",
      "speaker_label": "Клиент",
      "baseline_text": "Да, слушаю.",
      "variant_a": "Да, слушаю.",
      "variant_b": "Да, слушаю.",
      "channel_source": "right",
      "flags": []
    }
  ],
  "quality_hints": {
    "same_ts_cross": 3,
    "warnings": [
      "stereo_sequence_fix: swapped_adjacent_pairs=2"
    ]
  }
}
```

### Комментарии к входному контракту

- `turn_id` обязателен и служит опорой для валидации;
- `baseline_text` — текущий лучший текст turn до LLM;
- `variant_a` и `variant_b` нужны для выбора лучшей формулировки;
- `speaker` и `ts_sec` считаются опорными полями, LLM не должна их свободно переписывать;
- `flags` можно использовать для передачи признаков риска: `same_ts`, `echo_candidate`, `artifact_candidate`.

## 7. Новый выходной JSON от Codex

Ожидаемый ответ:

```json
{
  "schema_version": "dialogue_resolve_result_v1",
  "turns": [
    {
      "turn_id": 1,
      "speaker": "manager",
      "ts_sec": 2.1,
      "final_text": "Здравствуйте, меня зовут Иван, учебный центр Фотон.",
      "selection": "A",
      "action": "keep",
      "swap_with_next": false,
      "drop": false,
      "confidence": 0.92,
      "notes": ""
    },
    {
      "turn_id": 2,
      "speaker": "client",
      "ts_sec": 3.4,
      "final_text": "Да, слушаю.",
      "selection": "MIX",
      "action": "keep",
      "swap_with_next": false,
      "drop": false,
      "confidence": 0.84,
      "notes": ""
    }
  ],
  "warnings": [],
  "global_notes": ""
}
```

### Ограничения выхода

- `turn_id` должен совпадать со входом;
- число turns должно совпадать со входом или отличаться только из-за `drop=true` на явном артефакте;
- `speaker` может быть изменен только при явной ошибке baseline и должен дополнительно логироваться;
- `swap_with_next=true` разрешен только для соседней пары;
- `confidence` используется только как вспомогательный сигнал, не как единственный критерий.

## 8. Prompt для Codex

LLM prompt должен быть жестким и коротким.

Черновой системный смысл:

1. Ты исправляешь последовательность реплик и качество текста в уже распознанном телефонном разговоре.
2. Используй только входные turns и тексты `baseline_text`, `variant_a`, `variant_b`.
3. Нельзя придумывать новые факты и новые turns.
4. Нельзя переставлять несоседние turns.
5. Если baseline корректен, не меняй turn без причины.
6. Если реплика выглядит как эхо/мусор, можно поставить `drop=true`.
7. Верни только strict JSON заданной схемы.

Отдельное требование:

- если не уверен, оставляй baseline;
- если обе версии сомнительны, бери более консервативную формулировку;
- исправляй только локальные, а не глобальные перестановки.

## 9. Валидация ответа LLM

После ответа Codex нужен жесткий валидатор.

### Проверки уровня схемы

- JSON parse;
- наличие `schema_version`;
- наличие массива `turns`;
- у каждого turn есть `turn_id`, `final_text`, `drop`, `swap_with_next`.

### Проверки структурной целостности

- все `turn_id` из входа присутствуют;
- нет новых `turn_id`;
- `swap_with_next` не образует длинных цепочек;
- удаление turn возможно только если текст короткий и артефактный;
- порядок timestamps остается монотонным после локальных swap.

### Проверки безопасности

- длина `final_text` не должна быть аномально больше max(`baseline`, `A`, `B`);
- нельзя, чтобы слишком много turns были переписаны полностью;
- если доля changed turns выше порога, candidate помечается как risky;
- при невалидном ответе LLM candidate отбрасывается, pipeline продолжает работать.

## 10. Новая модель scoring

Текущее слабое место: LLM candidate не должен скориться через старый export-файл.

Новая логика:

- каждый candidate обязан нести свои `dialogue_lines`;
- scoring работает только по dialogue_lines самого candidate;
- fallback в старый `_text.txt` разрешен только для baseline, но не для LLM candidate.

### Метрики score

- `same_ts_cross`
- `speaker_flip_density`
- `role_length_balance`
- `artifact_lines`
- `dropped_turns`
- `swap_count`
- `few_words`
- `coverage_vs_baseline`
- `warnings_count`

Смысл:

- локальные swaps допустимы, но штрафуются, если их слишком много;
- небольшое количество `drop` допустимо, если это убирает эхо/мусор;
- агрессивное переписывание должно снижать score.

## 11. Что менять в коде

### 11.1 `src/mango_mvp/services/transcribe.py`

Оставить:

- текущий baseline merge;
- текущую сборку `dialogue_lines`;
- текущий export как fallback.

Добавить:

- builder `dialogue_resolve_payload` на основе текущих `dialogue_lines`, `variant_a`, `variant_b`, `merge_meta`, сегментов и ролей;
- enrich turns полями `turn_id`, `ts_sec`, `speaker`, `baseline_text`, `variant_a`, `variant_b`, `flags`.

### 11.2 `src/mango_mvp/services/resolve.py`

Основная переделка.

Добавить:

- `_build_dialogue_resolve_payload(call, variants_payload, baseline_dialogue_lines)`
- `_resolve_dialogue_with_codex(call, payload)`
- `_validate_dialogue_resolve_result(input_payload, llm_payload)`
- `_candidate_from_dialogue_result(...)`
- `_dialogue_lines_to_role_texts(...)`

Изменить:

- `_resolve_with_llm(...)`
- для `stereo` вместо per-role merge вызвать new dialogue-level resolve;
- candidate должен возвращать не `dialogue_lines=None`, а полноценный список lines.

Исправить:

- scoring path не должен использовать старый export для LLM candidate.

### 11.3 `src/mango_mvp/services/analyze.py`

Изменить вход:

- при наличии финальных `dialogue_lines` анализ строится именно по ним;
- `evidence` извлекается из финальных dialogue turns;
- `history_summary` должен ссылаться на реальные turn-level факты.

### 11.4 `src/mango_mvp/gui.py`

Добавить:

- выбор режима `Resolve mode`:
  - `legacy per-role`
  - `dialogue-level`
- отдельный флаг:
  - `Полный dialogue-resolve через Codex`
- понятное описание в UI, что этот режим медленнее, но качественнее.

## 12. Rollout plan

### Этап 1. Исправление инфраструктуры

- убрать scoring fallback для LLM candidate;
- научить LLM candidate возвращать реальные `dialogue_lines`;
- не менять пока весь prompt.

### Этап 2. Новый prompt и schema

- добавить новый dialogue-level prompt;
- добавить валидатор ответа;
- подключить новую стадию в `Resolve`.

### Этап 3. Пилот

- прогон 50 звонков;
- ручная оценка очередности реплик;
- сравнение:
  - baseline
  - old role-LLM
  - new dialogue-LLM
  - rescue-ASR

### Этап 4. Выбор default режима

- если прирост качества заметный, сделать `dialogue-level resolve` default для HQ режима;
- если прирост умеренный, оставить его только для risky calls.

## 13. Метрики успеха

Новый режим считается успешным, если на пилоте:

- уменьшается число явных ошибок очередности реплик;
- уменьшается число звонков с блоковым текстом вместо timed-dialogue;
- растет качество извлечения `Analyze`;
- LLM начинает выигрывать у baseline не эпизодически, а стабильно на risky subset.

## 14. Практическое решение по режимам

Предлагается ввести 2 рабочих режима.

### Fast

- `Whisper/MLX`
- `GigaAM`
- baseline resolve
- selective rescue
- Analyze

Для больших ночных прогонов.

### HQ Dialogue

- `Whisper/MLX`
- `GigaAM`
- dialogue-level Codex resolve
- rescue-ASR только если итоговый score ниже порога
- Analyze

Для финального слоя качества и для CRM-подготовки.

## 15. Следующий шаг реализации

Первый практический шаг реализации:

1. исправить scoring bug для LLM candidate;
2. реализовать `dialogue_resolve_payload`;
3. добавить новый prompt и parser;
4. научить `Resolve` возвращать собственные `dialogue_lines`.

Это минимальный кусок, после которого появится уже реальный measurable pilot.
