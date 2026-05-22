# Gold Answer Candidates

Gold-кандидат здесь означает “полезный шаблон мышления и качества ответа”. Это не готовый клиентский текст и не утвержденный факт.

| candidate_id | brand | theme | candidate | status | required_before_use |
|---|---|---|---|---|---|
| gold_001 | cross_brand_risk | answer_structure | прямой ответ первым, затем один безопасный следующий вопрос | style_candidate | проверить на шаблонность |
| gold_002 | cross_brand_risk | no_fact | если факта нет, дать безопасную рамку и передать точную проверку менеджеру | rule_candidate | добавить usefulness gate |
| gold_003 | foton | format_choice | объяснять онлайн/очно через цель ребенка, дорогу, записи и нагрузку | style_candidate | брать факты только из current KB |
| gold_004 | unpk | format_choice | объяснять онлайн/очно без сравнения с Фотоном | style_candidate | brand gate |
| gold_005 | foton | installment | рассрочка только из утвержденного gold/current KB | needs_kb_fix | убрать мусорный срок из current KB |
| gold_006 | unpk | installment | не обещать банковскую рассрочку и чужие сервисы | covered_by_current_gold | regression test |
| gold_007 | cross_brand_risk | camps | спрашивать класс, не возраст; места не обещать | covered_by_current_gold | fresh camp source |
| gold_008 | unpk | payment_status | по оплате отвечать общей рамкой, статус только менеджеру | manager_only | read-only AMO/Tallanto evidence |
| gold_009 | cross_brand_risk | documents | справки и документы: общий порядок отдельно, индивидуальный статус отдельно | needs_rop_approval | владелец документа |
| gold_010 | cross_brand_risk | tax_matkap | решение принимает госорган, центр помогает с документами | covered_by_current_gold | не обещать одобрение |
| gold_011 | cross_brand_risk | access | доступы и ссылки: признать проблему, попросить минимум данных, передать проверку | manager_draft_only | recipient/identity guard |
| gold_012 | cross_brand_risk | composite_question | если есть P0-часть, общий route не автономный; безопасную часть сохранить в черновике | rule_candidate | tests |
| gold_013 | unpk | waiting_list | лист ожидания не должен звучать как гарантированная запись | needs_kb_fix | route draft/manager |
| gold_014 | cross_brand_risk | historical_style | исторический хороший ответ можно брать как стиль, не как дословный шаблон | rule_candidate | anti-verbatim gate |
