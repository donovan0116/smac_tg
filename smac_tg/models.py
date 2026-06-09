from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TacticalCommand:
    tactic: str
    priority_target: str | None = None
    confidence: float = 0.0
    reason: str = ""
    raw_answer: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def hold_position(reason: str = "No TrustGraph decision available.") -> "TacticalCommand":
        return TacticalCommand(
            tactic="hold_position",
            priority_target=None,
            confidence=0.0,
            reason=reason,
        )

