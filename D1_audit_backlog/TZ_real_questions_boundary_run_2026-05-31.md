# ТЗ MAIN (главный Кодекс) — прогон бота на РЕАЛЬНЫХ вопросах клиентов → граничные случаи. 2026-05-31.

Автор: Клод 1. Цель — найти слепые зоны/граничные ответы на НАСТОЯЩИХ формулировках клиентов (не
синтетических персонах). У главного Кодекса локально есть `whatsapp_chats.sqlite` с реальными диалогами
(у Claude #1 она пустая-заглушка из-за приватности). Кодекс извлекает, прогоняет, возвращает граничные.

## ВАЖНО про приватность

База содержит ПДн (телефоны, ФИО, email, адреса). При извлечении вопросов — МАСКИРОВАТЬ: телефоны
`\+?\d[\d\s\-()]{8,}` → «[телефон]», email → «[email]», явные ФИО/имена детей → «[имя]». Сами ПДн
наружу/в набор НЕ выносить. Итоговый набор вопросов — без персональных данных.

## ШАГ 1 — извлечь реальные клиентские вопросы из БД

Из `product_data/transcripts/whatsapp_chats.sqlite`, таблица `messages` (роль клиента, не служебные).
Ориентир SQL (адаптируй под реальные имена колонок — посмотри `PRAGMA table_info(messages)`):
```sql
SELECT text, brand_hint FROM messages
WHERE role LIKE '%lient%' AND is_service_message = 0
  AND text LIKE '%?%' AND length(text) BETWEEN 15 AND 160;
```
Отбери ~150-200 РАЗНООБРАЗНЫХ вопросов: по темам (цена, формат, расписание, лагерь, олимпиады, документы/
справки/вычет, скидки, оплата частями, пробное) и по обоим брендам (по brand_hint, foton/unpk; mixed/null
не приписывать). Внутри темы бери разные формулировки (живые, с опечатками, разговорные — в этом ценность).
Дедуплицируй близкие. Замаскируй ПДн.

## ШАГ 2 — превратить в набор персон (одноходовые)

Формат — как в существующих наборах (persona + simulator_spec + judge_spec; переиспользуй спеки из
`targeted_pravka5.jsonl`). Для каждого реального вопроса — persona:
```json
{"type":"persona","brand":"<foton|unpk по brand_hint>","category":"real_question",
 "dialog_id":"real_<тема>_<n>","max_turns":2,"expected_route":"bot_answer_self",
 "persona":"real client question","goal":"<тема>","held_facts":{},
 "behaviors":["<реальный замаскированный вопрос>"],
 "success_criteria":"ответил по существу из факта, если факт есть; ушёл к менеджеру только если факта нет/P0",
 "fail_criteria":"ушёл к менеджеру при наличии факта; выдумал; смешал бренды",
 "brand_forbidden":[<маркеры другого бренда>]}
```
Сохрани как `product_data/telegram_dynamic_test_sets/real_questions_20260531.jsonl`.

## ШАГ 3 — прогнать бота (v2 ON + semantic, parallel 4)

```bash
# временно в раннер: "allow_default_autonomy": True,  (после — убрать)
TELEGRAM_DIALOGUE_CONTRACT_PIPELINE=1 \
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/run_telegram_dynamic_client_sim.py \
  --scenarios product_data/telegram_dynamic_test_sets/real_questions_20260531.jsonl \
  --snapshot <v6.4 snapshot> \
  --memory-mode off --semantic-mode codex --semantic-reasoning medium --parallel 4 \
  --out-dir runs/20260531_real_questions
```
(memory off — это одноходовые вопросы, память не нужна; parallel 4 — правило замеров.)

## ШАГ 4 — вернуть ГРАНИЧНЫЕ случаи (главный результат)

Не интерпретируй судью — собери ФАКТЫ для Claude #1. По транскриптам выдели и пришли списком:
- **Ушёл к менеджеру при наличии факта** (route=draft_for_manager/manager_only ИЛИ текст-хендофф, но
  retrieved_facts непусто): dialog_id, вопрос клиента, какие факты были в retrieved, текст бота. ← это
  главные граничные случаи (где факт есть, но бот не связал — кандидаты на уточнение фактов).
- **Выдумки** (hard_gate fabrication): вопрос, факт, текст.
- **Темы, где бот часто уходит** (агрегат: по каким темам больше всего хендоффов при фактах).
- Сводка: PASS/PWN/FAIL, hard_gate, автономия, бренд/мета (должно 0/0), infra.

## Ограничения

- ПДн замаскированы везде (набор, отчёт). Сам набор — без персональных данных, можно в Яндекс/repo.
- Не менять код бота. Прогон на parallel 4. Это разведка слепых зон, не правка.
- Результаты (набор + runs + список граничных) — в Яндекс, чтобы Claude #1 разобрал по сырью.
