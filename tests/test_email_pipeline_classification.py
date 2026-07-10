from scripts.email_pipeline.classification import ClassificationInput, OWN_DOMAINS, classify_message
from scripts.email_pipeline.orphan_driver import owner_bucket


def _msg(**overrides):
    base = dict(
        kind="normal",
        mailbox="INBOX",
        from_email="parent@example.com",
        from_dom="example.com",
        from_local="parent",
        to_doms=("kmipt.ru",),
        subject="Вопрос по курсу",
        body_chars=120,
        eml_flags={"list_unsub": False, "bulk": False, "auto": False, "campaign": False},
        is_outbound=False,
    )
    base.update(overrides)
    return ClassificationInput(**base)


def test_inbound_human_is_real_correspondence() -> None:
    assert classify_message(_msg(), set()) == ("real_correspondence", "inbound_human")


def test_list_unsubscribe_is_bulk_newsletter() -> None:
    result = classify_message(_msg(eml_flags={"list_unsub": True, "bulk": False, "auto": False, "campaign": False}), set())
    assert result[0] == "bulk_newsletter"


def test_outbound_template_is_campaign_not_real_correspondence() -> None:
    result = classify_message(
        _msg(
            from_email="edu@kmipt.ru",
            from_dom="kmipt.ru",
            from_local="edu",
            to_doms=("example.com",),
            subject="Расписание занятий",
            is_outbound=True,
        ),
        {"расписание занятий"},
    )
    assert result == ("outbound_campaign", "template_mass_send")


def test_own_domains_match_mail_link_enrich_scope() -> None:
    assert {"kmipt.ru", "cdpofoton.ru", "foton.school", "amocrm.ru", "amocrm.com"} <= OWN_DOMAINS


def test_own_to_own_across_foton_domains_is_internal() -> None:
    result = classify_message(
        _msg(
            from_email="edu@cdpofoton.ru",
            from_dom="cdpofoton.ru",
            from_local="edu",
            to_doms=("foton.school", "amocrm.ru"),
            is_outbound=True,
        ),
        set(),
    )

    assert result == ("internal", "own_to_own")


def test_orphan_owner_bucket_keeps_review_needed_real_mail_for_manager_review() -> None:
    assert owner_bucket("real_correspondence", source_kind="llm") == "real_candidate"
    assert owner_bucket("real_correspondence", source_kind="llm_review_needed") == "real_manager_review"
    assert owner_bucket("bulk_newsletter", source_kind="llm") == "broadcast_or_internal_skip"
