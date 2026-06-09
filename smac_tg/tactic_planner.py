from __future__ import annotations

from typing import Any

from .config import TrustGraphSMACConfig
from .models import TacticalCommand
from .state_encoder import SMACStateEncoder
from .trustgraph_client import TrustGraphTacticalClient


class TrustGraphTacticPlanner:
    """Orchestrates state upload, Graph RAG query, and JSON normalization."""

    def __init__(self, config: TrustGraphSMACConfig):
        self.config = config
        self.encoder = SMACStateEncoder(config)
        self.client = TrustGraphTacticalClient(config)

    def plan(self, episode_id: str, step: int, obs_summary: dict[str, Any]) -> TacticalCommand:
        triples = self.encoder.encode(episode_id, step, obs_summary)
        self.client.import_state_triples(episode_id, step, triples)

        question = build_tactical_question(episode_id, step, obs_summary)
        answer = self.client.ask_graph_rag(question)
        return self.client.normalize_tactic_json(answer)


def build_tactical_question(episode_id: str, step: int, obs_summary: dict[str, Any]) -> str:
    return f"""
Episode: {episode_id}
Step: {step}
Map: {obs_summary.get("map_name", "unknown")}

Given the current SMAC battle graph, choose one high-level tactic for allied units:
focus_fire, kite, retreat, regroup, flank, hold_position, or push.

Return the best tactical option, the priority enemy target if any, the key reason,
and whether low-health allies or low-health enemies dominate the decision.
"""

