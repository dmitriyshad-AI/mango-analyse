from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class ReplayMessage:
    profile_id: str
    chat_id: str
    message_id: str
    text: str
    timestamp: int
    from_me: bool
    ts_masked: str = ""
    sender_name: str = ""
    raw: Mapping[str, Any] = field(default_factory=dict)

    @property
    def is_client(self) -> bool:
        return bool(self.text.strip()) and not self.from_me

    @property
    def is_manager(self) -> bool:
        return bool(self.text.strip()) and self.from_me


@dataclass(frozen=True)
class ReplayCase:
    dialog_id: str
    profile_id: str
    chat_id: str
    turn_id: str
    brand: str
    client_message: str
    manager_reference: str
    turn_index: int = 0
    contour: str = ""
    dialog_key_masked: str = ""
    prefix_messages: tuple[ReplayMessage, ...] = ()
    segment: str = "chat_only"
    expected_p0: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BotReplayResult:
    route: str
    bot_text: str
    safety_flags: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
