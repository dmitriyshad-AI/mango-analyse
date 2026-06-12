#!/usr/bin/env python3
"""Для каждого кластера (кроме P0) находит КАНДИДАТ-факты базы по теме, по брендам.
Выход: _coverage_candidates.json — вход субагентам для присвоения статуса покрытия."""
import json, re, collections
base="D1_audit_backlog/kb_intake_20260610/"
facts=json.load(open(base+"_kb_client_facts.json",encoding="utf-8"))
scan=json.load(open(base+"_questions_scan.json",encoding="utf-8"))

KW={
 "F_price":["цена","стоим","руб","₽","стоит","прайс","оплат","price"],
 "F_schedule":["расписан","занятия проход","выходн","воскресен","суббот","во сколько","weekly","schedule","раз в недел","день недел"],
 "F_address":["адрес","локац","location","метро","корпус","площадк","ул.","проспект","шоссе"],
 "F_format":["онлайн","очно","формат","дистанц","платформ"],
 "F_programs":["предмет","программ","направлен","математик","физик","информатик","русск","subjects","курс"],
 "F_subject_avail":["хими","английск","робототехник","биолог","unavailable","не запуска","нет ","предмет"],
 "F_teachers":["преподавател","педагог","teacher","регали","опыт"],
 "F_discounts":["скидк","discount","льгот","многодетн","second_subject","акци","кэшб","loyal"],
 "F_duration":["длительн","ак.ч","ак. час","total_lessons","weekly","часов","минут","занятие длит"],
 "F_age_grade":["класс","возраст","grade","с какого","лет","audience"],
 "F_group_size":["групп","group_size","человек","наполн"],
 "F_proof":["результат","поступ","social_proof","alumni","winner","olymp"],
 "F_olymp":["олимпиад","физтех","перечнев","росатом","курчатов","всош","оммо"],
 "F_start":["старт","начал","academic_year","сентябр","начин"],
 "F_camp_info":["лагер","лвш","смен","лш","городск","camp","ls_city","lvsh","менделеево","выездн"],
 "F_lvsh_logistics":["прожив","питан","трансфер","accommodation","transfer","шведск","номер","медсестр"],
 "F_deadline":["срок","payment_deadline","до какого","запис"],
 "P_enroll":["запис","оформ","enroll","процесс","зачислен"],
 "P_after_pay":["после оплат","доступ","ссылк","приглаш"],
 "P_pay_how":["оплат","payment","qr","квитанц","реквизит","способ"],
 "P_installment":["рассроч","installment","долями","помесячн","payment_options","частями"],
 "P_check":["чек","квитанц","подтвержд","оплат прош"],
 "P_contract":["договор","оферт","contract"],
 "P_taxcert":["вычет","справк","tax_deduction","ндфл","кнд","фнс"],
 "P_matkapital":["маткапитал","материнск","matkap","сфр"],
 "P_test_how":["тест","placement","вступительн","входн","диагностик"],
 "P_test_results":["результат тест","итог тест","распредел"],
 "P_group_assign":["групп","уровень","базов","продвинут","олимпиадн","level","распредел","деление"],
 "P_link":["ссылк","платформ","мтс","mts","link","подключ","webinar","вебинар"],
 "P_cabinet":["кабинет","пароль","логин","account","личн"],
 "P_records":["запис заняти","материал","recording","записи урок"],
 "P_transfer_freeze":["перенос","заморозк","перенест","пауз"],
 "P_missed":["пропуск","missed","болезн","отработ","095","справк"],
 "P_waitlist":["лист ожидан","ожидан","waitlist","zvsh"],
 "P_certificate":["сертификат","certificate","свидетельств","удостоверен","об окончан"],
 "P_techissue":["спам","gmail","не пришл","техническ","настройк","поддержк"],
 "P_promocode":["промокод","promo","акци"],
 "P_change_group":["перевод","сменить групп","level_transfer","друг групп","формат"],
 "P_trial":["пробн","trial","фрагмент","демо"],
 "P_contact":["контакт","телефон","перезвон","contact","hotline","горяч","email","связ"],
 "P_foreign":["иностран","казахстан","заграниц","другой стран","региональн"],
 "P_referral":["приведи друг","refer_a_friend","кэшб","привед"],
 "A_recommend":["подбор","посовет","рекоменд","уровень","выбрать","levels_explained"],
}

def cand(cid, brand):
    kws=KW.get(cid,[])
    scored=[]
    for f in facts:
        if f.get("brand")!=brand: continue
        blob=((f.get("fact_key") or "")+" "+(f.get("text") or "")).lower()
        sc=sum(1 for k in kws if k in blob)
        if sc>0: scored.append((sc,f))
    scored.sort(key=lambda x:-x[0])
    return [{"fact_key":f["fact_key"],"text":f["text"][:280]} for _,f in scored[:6]]

out=[]
for c in scan["clusters"]:
    if c["class"]=="P0": continue
    out.append({"cid":c["cid"],"class":c["class"],"label":c["label"],
                "wa":c["wa"],"tg":c["tg"],"call":c["call"],
                "foton_candidates":cand(c["cid"],"foton"),
                "unpk_candidates":cand(c["cid"],"unpk")})
json.dump(out,open(base+"_coverage_candidates.json","w",encoding="utf-8"),ensure_ascii=False,indent=1)
print("кластеров (не P0):",len(out))
# быстрый обзор: сколько кластеров без кандидатов по бренду
no_f=[o["cid"] for o in out if not o["foton_candidates"]]
no_u=[o["cid"] for o in out if not o["unpk_candidates"]]
print("без кандидатов Фотон:",no_f)
print("без кандидатов УНПК:",no_u)
