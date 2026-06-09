from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class TrustGraphSMACConfig:
    url: str = os.getenv("TRUSTGRAPH_URL", "http://localhost:8088/")
    token: str | None = os.getenv("TRUSTGRAPH_TOKEN")
    workspace: str = os.getenv("TRUSTGRAPH_WORKSPACE", "default")
    flow_id: str = "smac-onto-rag"
    collection: str = "smac"
    ontology_namespace: str = "https://example.org/ontology/smac#"
    instance_namespace: str = "https://example.org/smac/"
    decision_interval: int = 40
    entity_limit: int = 20
    triple_limit: int = 10
    max_subgraph_size: int = 80
    max_path_length: int = 2
    edge_score_limit: int = 20
    edge_limit: int = 10
    enable_trustgraph: bool = True

