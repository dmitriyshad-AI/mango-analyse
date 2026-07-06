from __future__ import annotations

from typing import Sequence

from .models import ReplayCase, ReplayMessage


def _segment_for_reference(reference: str) -> str:
    lowered = reference.casefold()
    if any(marker in lowered for marker in ("позвон", "crm", "амосрм", "талланто", "tallanto", "проверю в системе")):
        return "external_context"
    if any(marker in lowered for marker in ("ошиб", "неправильно", "лучше так", "менеджер")):
        return "manager_issue_private"
    return "chat_only"


def slice_teacher_forcing_cases(
    messages: Sequence[ReplayMessage],
    *,
    dialog_id: str,
    brand: str,
    manager_window_seconds: int = 600,
    burst_seconds: int = 120,
) -> list[ReplayCase]:
    ordered = sorted(messages, key=lambda item: (item.timestamp, item.message_id))
    cases: list[ReplayCase] = []
    index = 0
    turn_no = 0
    while index < len(ordered):
        current = ordered[index]
        if not current.is_client:
            index += 1
            continue
        burst = [current]
        cursor = index + 1
        while (
            cursor < len(ordered)
            and ordered[cursor].is_client
            and ordered[cursor].timestamp - burst[-1].timestamp <= burst_seconds
        ):
            burst.append(ordered[cursor])
            cursor += 1
        next_client_ts = None
        for future in ordered[cursor:]:
            if future.is_client:
                next_client_ts = future.timestamp
                break
        manager_refs: list[ReplayMessage] = []
        for future in ordered[cursor:]:
            if not future.is_manager:
                continue
            if next_client_ts is not None and future.timestamp >= next_client_ts:
                break
            if future.timestamp - burst[-1].timestamp <= manager_window_seconds:
                manager_refs.append(future)
        if manager_refs:
            turn_no += 1
            client_message = "\n".join(item.text for item in burst)
            manager_reference = "\n".join(item.text for item in manager_refs)
            prefix = tuple(item for item in ordered[:index] if item.text.strip())
            cases.append(
                ReplayCase(
                    dialog_id=dialog_id,
                    profile_id=current.profile_id,
                    chat_id=current.chat_id,
                    turn_id=f"{dialog_id}#{turn_no}",
                    brand=brand,
                    client_message=client_message,
                    manager_reference=manager_reference,
                    prefix_messages=prefix,
                    segment=_segment_for_reference(manager_reference),
                    metadata={"burst_size": len(burst), "manager_reference_count": len(manager_refs)},
                )
            )
        index = cursor
    return cases
