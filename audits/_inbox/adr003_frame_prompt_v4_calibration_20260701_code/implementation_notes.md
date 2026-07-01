# ADR-003 Frame Prompt v4 Calibration

Дата: 2026-07-01.

Задача: Ф1 цели максимальной автономности через SemanticFrame. Исправить schema drift `answerability=yes/no` и откалибровать post-hoc SemanticFrame так, чтобы он меньше путал безопасную справку с manager-only действиями, не меняя route/text.

Изменения:

- `provider.py`: отдельная нормализация `semantic_frame.answerability` в enum `answer_self|manager_only|uncertain`.
- `provider.py`: post-hoc prompt уточняет границу safe reference vs manager action:
  - safe self только если весь запрос является справкой и уже есть проверенный факт;
  - деньги, сроки/порядок оплаты, чек, рассрочка, предоплата, списания/удержания/возвратные условия, запись/бронь/места/группа, индивидуальная ситуация ребёнка остаются manager-only.
- Тесты: добавлены проверки enum-нормализации и ключевых границ prompt; старый semantic-frame shadow тест обновлён с legacy `yes` на новый `answer_self`.
- Документация: `docs/ADR003_SEMANTIC_FRAME_EVAL.md` дополнен сравнением initial/v2/v3/v4.

Локальные сырые отчёты:

- `audits/_inbox/adr003_frame_prompt_v4_calibration_20260701/adr003_frame_gold_calibration_report.json`
- `audits/_inbox/adr003_frame_prompt_v4_calibration_20260701/adr003_frame_gold_calibration_report.md`
- `audits/_inbox/adr003_frame_prompt_v4_calibration_20260701/adr003_semantic_frame_eval_report.json`
- `audits/_inbox/adr003_frame_prompt_v4_calibration_20260701/adr003_semantic_frame_eval_report.md`

