from __future__ import annotations

from typing import Any, Iterable

from trustgraph.api import Triple

from .config import TrustGraphSMACConfig

RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"


class SMACStateEncoder:
    """Convert a normalized SMAC observation summary into RDF triples."""

    def __init__(self, config: TrustGraphSMACConfig):
        self.config = config

    def encode(self, episode_id: str, step: int, obs_summary: dict[str, Any]) -> list[Triple]:
        ns = self.config.ontology_namespace
        inst = self._episode_namespace(episode_id)
        battle = f"{inst}BattleState-{step}"
        triples: list[Triple] = [
            Triple(battle, RDF_TYPE, f"{ns}BattleState"),
            Triple(battle, f"{ns}timeStep", str(step)),
        ]

        map_name = obs_summary.get("map_name")
        if map_name:
            triples.append(Triple(battle, f"{ns}mapName", str(map_name)))

        for unit in obs_summary.get("allies", []):
            unit_uri = self._unit_uri(inst, unit)
            triples.extend(self._unit_triples(unit_uri, unit, f"{ns}AllyUnit"))
            triples.append(Triple(battle, f"{ns}hasAllyUnit", unit_uri))

        for unit in obs_summary.get("enemies", []):
            unit_uri = self._unit_uri(inst, unit)
            triples.extend(self._unit_triples(unit_uri, unit, f"{ns}EnemyUnit"))
            triples.append(Triple(battle, f"{ns}hasEnemyUnit", unit_uri))

        triples.extend(self._relation_triples(inst, obs_summary.get("relations", [])))
        triples.extend(self._derived_tactical_facts(inst, obs_summary))

        return triples

    def _episode_namespace(self, episode_id: str) -> str:
        return f"{self.config.instance_namespace}{safe_id(episode_id)}#"

    def _unit_uri(self, inst: str, unit: dict[str, Any]) -> str:
        return f"{inst}{safe_id(unit.get('id', 'unknown-unit'))}"

    def _unit_triples(self, unit_uri: str, unit: dict[str, Any], rdf_class: str) -> list[Triple]:
        ns = self.config.ontology_namespace
        triples = [Triple(unit_uri, RDF_TYPE, rdf_class)]

        field_map = {
            "unit_type": "unitType",
            "health": "health",
            "shield": "shield",
            "energy": "energy",
            "x": "x",
            "y": "y",
            "weapon_cooldown": "weaponCooldown",
            "is_alive": "isAlive",
        }

        for source_key, predicate in field_map.items():
            if source_key in unit and unit[source_key] is not None:
                triples.append(Triple(unit_uri, f"{ns}{predicate}", str(unit[source_key])))

        return triples

    def _relation_triples(self, inst: str, relations: Iterable[dict[str, Any]]) -> list[Triple]:
        ns = self.config.ontology_namespace
        triples: list[Triple] = []

        for relation in relations:
            subject = relation.get("subject")
            predicate = relation.get("predicate")
            obj = relation.get("object")
            if not subject or not predicate or not obj:
                continue

            subject_uri = f"{inst}{safe_id(subject)}"
            object_uri = f"{inst}{safe_id(obj)}"
            triples.append(Triple(subject_uri, f"{ns}{predicate}", object_uri))

            if "distance" in relation:
                rel_id = f"{inst}Relation-{safe_id(subject)}-{safe_id(predicate)}-{safe_id(obj)}"
                triples.append(Triple(rel_id, RDF_TYPE, f"{ns}SpatialRelation"))
                triples.append(Triple(rel_id, f"{ns}relationSubject", subject_uri))
                triples.append(Triple(rel_id, f"{ns}relationObject", object_uri))
                triples.append(Triple(rel_id, f"{ns}distance", str(relation["distance"])))

        return triples

    def _derived_tactical_facts(self, inst: str, obs_summary: dict[str, Any]) -> list[Triple]:
        ns = self.config.ontology_namespace
        triples: list[Triple] = []

        low_health_enemies = [
            enemy for enemy in obs_summary.get("enemies", [])
            if enemy.get("is_alive", True) and float(enemy.get("health", 9999)) <= 25.0
        ]
        if low_health_enemies:
            target = min(low_health_enemies, key=lambda item: float(item.get("health", 9999)))
            target_uri = self._unit_uri(inst, target)
            threat_uri = f"{inst}PriorityTarget-{safe_id(target.get('id', 'unknown'))}"
            triples.append(Triple(threat_uri, RDF_TYPE, f"{ns}PriorityTarget"))
            triples.append(Triple(threat_uri, f"{ns}targetUnit", target_uri))
            triples.append(Triple(threat_uri, f"{ns}targetReason", "Enemy unit has low health."))

        low_health_allies = [
            ally for ally in obs_summary.get("allies", [])
            if ally.get("is_alive", True) and float(ally.get("health", 9999)) <= 20.0
        ]
        if low_health_allies:
            risk_uri = f"{inst}LowHealthAllyRisk"
            triples.append(Triple(risk_uri, RDF_TYPE, f"{ns}RetreatRisk"))
            for ally in low_health_allies:
                triples.append(Triple(risk_uri, f"{ns}riskUnit", self._unit_uri(inst, ally)))

        return triples


def safe_id(value: Any) -> str:
    text = str(value)
    return "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in text)

