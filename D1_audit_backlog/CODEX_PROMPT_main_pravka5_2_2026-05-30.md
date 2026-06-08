# Стартовый промт — MAIN Кодекс (правка 5.2: P0b-жалоба). 2026-05-30.

> Скопировать в окно MAIN. Идёт в ОДИН батч с 5.1 — после обеих один прогон M1.

---

Ты — MAIN Кодекс, HEAD `6dd7cf2a` (после правки 5.1).

**Прочитай:** `D1_audit_backlog/TZ_pravka5_2_p0_complaint_2026-05-30.md` — там разбор и готовые сниппеты.

**Проблема (подтверждена чтением кода):** на жалобе маршрут `manager_only` верный, но текст плохой —
бот выдаёт справку про скидку (unpk) или просит данные ребёнка (foton). Причина: `_safe_fallback_text`
не различает жалобу и уходит в `_secondary_fact_text` / «уточнить деталь». Флаг `zero_collect_required`
для жалобы уже вычисляется в `classify_answer_safety`, но нигде не читается.

**Правка 1:** добавить `_COMPLAINT_HANDOFF_TEXTS` + `_complaint_handoff_text()` рядом с
`_refund_policy_handoff_text` (~2165) — готовый текст в ТЗ. Эмпатия + чистый хендофф, без сбора данных,
без обещаний, без продаж.

**Правка 2:** в начале `_safe_fallback_text` (~2179, после `traced`, ДО `known_absence`/secondary/detail)
вставить zero_collect-ветку (готовый сниппет в ТЗ): complaint → `_complaint_handoff_text`, refund →
`_refund_policy_handoff_text`, прочее P0 → сухой generic. `classify_answer_safety` уже импортирован.

**Не трогать:** `_secondary_fact_text`/`question_detail`/`generic` для НЕ-P0; COMPLAINT_RE НЕ расширять
(детекция мягких жалоб — отдельный кандидат, не эта правка).

**Тесты:** complaint → handoff без «скидк/укажите/ребён/цифр»; refund → refund-handoff; НЕГАТИВНЫЙ
КОНТРОЛЬ — обычный вопрос (zero_collect=False) → secondary/detail работают как раньше.

**Ограничения:** тесты не гонять (лимит) — Клод 1 прогонит pipeline + smoke. В main не мержить до
зелёного. Правило #1: места и сигнатуры подтвердить чтением. Отчёт: что изменено, тексты тестов.