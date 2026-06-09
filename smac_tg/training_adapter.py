from __future__ import annotations

import logging
from typing import Any

from .config import TrustGraphSMACConfig
from .models import TacticalCommand
from .tactic_planner import TrustGraphTacticPlanner

logger = logging.getLogger(__name__)


class TrustGraphTrainingAdapter:
    """Stateful adapter intended to be called from a normal SMAC training loop."""

    def __init__(self, config: TrustGraphSMACConfig):
        self.config = config
        self._last_command = TacticalCommand.hold_position("No tactical query has been made yet.")
        self._planner = TrustGraphTacticPlanner(config) if config.enable_trustgraph else None

    @property
    def last_command(self) -> TacticalCommand:
        return self._last_command

    def maybe_update_tactic(
        self,
        episode_id: str,
        step: int,
        obs_summary: dict[str, Any],
    ) -> TacticalCommand:
        if not self.config.enable_trustgraph:
            return self._last_command

        if step % self.config.decision_interval != 0:
            return self._last_command

        if self._planner is None:
            return self._last_command

        try:
            self._last_command = self._planner.plan(
                episode_id=episode_id,
                step=step,
                obs_summary=obs_summary,
            )
        except Exception as exc:
            logger.exception("TrustGraph tactical planning failed")
            self._last_command = TacticalCommand.hold_position(
                reason=f"TrustGraph tactical planning failed: {exc}"
            )

        return self._last_command

    def encode_for_policy(self) -> dict[str, Any]:
        """Return a policy-friendly structure. Convert to tensor/one-hot in your code."""
        return {
            "tactic": self._last_command.tactic,
            "priority_target": self._last_command.priority_target,
            "confidence": self._last_command.confidence,
        }

