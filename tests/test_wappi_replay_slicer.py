from __future__ import annotations

from mango_mvp.replay_exam.models import ReplayMessage
from mango_mvp.replay_exam.slicer import slice_teacher_forcing_cases


def _msg(mid: str, text: str, ts: int, *, from_me: bool) -> ReplayMessage:
    return ReplayMessage(profile_id="p", chat_id="c", message_id=mid, text=text, timestamp=ts, from_me=from_me)


def test_slicer_merges_client_burst_and_attaches_manager_reference() -> None:
    cases = slice_teacher_forcing_cases(
        [
            _msg("1", "Здравствуйте", 10, from_me=False),
            _msg("2", "Нужна физика", 40, from_me=False),
            _msg("3", "Добрый день, расскажу про формат занятий и условия записи.", 100, from_me=True),
        ],
        dialog_id="d",
        brand="foton",
    )

    assert len(cases) == 1
    assert cases[0].client_message == "Здравствуйте\nНужна физика"
    assert cases[0].manager_reference == "Добрый день, расскажу про формат занятий и условия записи."
    assert cases[0].metadata["burst_size"] == 2
    assert cases[0].segment == "chat_only"


def test_slicer_marks_external_context_segment() -> None:
    cases = slice_teacher_forcing_cases(
        [
            _msg("1", "Оплатили, где доступ?", 10, from_me=False),
            _msg("2", "Проверю в системе оплату и вернусь с ответом по доступу.", 20, from_me=True),
        ],
        dialog_id="d",
        brand="unpk",
    )
    assert cases[0].segment == "external_context"


def test_slicer_marks_expected_p0_from_client_text() -> None:
    cases = slice_teacher_forcing_cases(
        [
            _msg("1", "Верните деньги за курс", 10, from_me=False),
            _msg("2", "Передам менеджеру, он проверит ситуацию и вернется с ответом.", 20, from_me=True),
        ],
        dialog_id="d",
        brand="foton",
    )

    assert cases[0].expected_p0 is True


def test_slicer_skips_short_manager_references() -> None:
    cases = slice_teacher_forcing_cases(
        [
            _msg("1", "Можно записаться?", 10, from_me=False),
            _msg("2", "Да", 20, from_me=True),
        ],
        dialog_id="d",
        brand="foton",
    )

    assert cases == []
