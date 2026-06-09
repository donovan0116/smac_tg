from __future__ import annotations

import json
import re
from collections.abc import Iterable

from trustgraph.api import Api, Triple

from .config import TrustGraphSMACConfig
from .models import TacticalCommand


class TrustGraphTacticalClient:
    """Thin wrapper around TrustGraph API calls used by the SMAC adapter."""

    def __init__(self, config: TrustGraphSMACConfig):
        self.config = config
        self.api = Api(
            url=config.url,
            token=config.token,
            workspace=config.workspace,
            timeout=60,
        )
        self.flow = self.api.flow().id(config.flow_id)
        self.bulk = self.api.bulk()

    def import_state_triples(self, episode_id: str, step: int, triples: Iterable[Triple]) -> None:
        self.bulk.import_triples(
            flow=self.config.flow_id,
            triples=iter(triples),
            metadata={
                "id": f"smac-{episode_id}-state-{step}",
                "metadata": [],
                "collection": self.config.collection,
            },
        )

    def ask_graph_rag(self, question: str) -> str:
        result = self.flow.graph_rag(
            query=question,
            collection=self.config.collection,
            entity_limit=self.config.entity_limit,
            triple_limit=self.config.triple_limit,
            max_subgraph_size=self.config.max_subgraph_size,
            max_path_length=self.config.max_path_length,
            edge_score_limit=self.config.edge_score_limit,
            edge_limit=self.config.edge_limit,
        )
        return result.text

    def normalize_tactic_json(self, graph_rag_answer: str) -> TacticalCommand:
        system = "Convert tactical analysis into strict JSON for a SMAC reinforcement-learning controller."
        prompt = f"""
Graph RAG tactical analysis:
{graph_rag_answer}

Return only JSON with this schema:
{{
  "tactic": "focus_fire|kite|retreat|regroup|flank|hold_position|push",
  "priority_target": "unit id or null",
  "confidence": 0.0,
  "reason": "short reason"
}}
"""
        result = self.flow.text_completion(system=system, prompt=prompt)
        return parse_tactical_command(result.text, raw_answer=graph_rag_answer)


def parse_tactical_command(text: str, raw_answer: str = "") -> TacticalCommand:
    payload_text = extract_json_object(text)
    try:
        payload = json.loads(payload_text)
    except Exception:
        return TacticalCommand.hold_position(
            reason=f"Failed to parse TrustGraph tactical JSON: {text[:200]}"
        )

    return TacticalCommand(
        tactic=str(payload.get("tactic") or "hold_position"),
        priority_target=payload.get("priority_target"),
        confidence=float(payload.get("confidence") or 0.0),
        reason=str(payload.get("reason") or ""),
        raw_answer=raw_answer,
        metadata={"llm_json": payload},
    )


def extract_json_object(text: str) -> str:
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return text

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found")
    return match.group(0)

