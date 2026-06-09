"""
Example integration point for a SMAC training loop.

This file intentionally does not implement SMAC training. Replace the TODO
sections with your existing PyMARL/SMAC/MARL code.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from smac_tg.config import TrustGraphSMACConfig
from smac_tg.training_adapter import TrustGraphTrainingAdapter


def main() -> None:
    tg_adapter = TrustGraphTrainingAdapter(
        TrustGraphSMACConfig(
            url=os.getenv("TRUSTGRAPH_URL", "http://localhost:8088/"),
            token=os.getenv("TRUSTGRAPH_TOKEN"),
            workspace=os.getenv("TRUSTGRAPH_WORKSPACE", "default"),
            flow_id=os.getenv("TRUSTGRAPH_FLOW_ID", "smac-onto-rag"),
            collection=os.getenv("TRUSTGRAPH_COLLECTION", "smac"),
            decision_interval=int(os.getenv("TG_DECISION_INTERVAL", "40")),
            enable_trustgraph=os.getenv("TG_ENABLED", "1") == "1",
        )
    )

    # TODO: Create your SMAC environment here.
    # env = StarCraft2Env(map_name="3m", ...)

    for episode_idx in range(1):
        episode_id = f"episode-{episode_idx}"

        # TODO: Reset your SMAC environment.
        # env.reset()

        for step in range(0, 200):
            # TODO: Replace this stub with a summary derived from real SMAC observations.
            obs_summary = build_stub_obs_summary(step)

            tactical_command = tg_adapter.maybe_update_tactic(
                episode_id=episode_id,
                step=step,
                obs_summary=obs_summary,
            )

            policy_context = tg_adapter.encode_for_policy()

            # TODO: Pass policy_context into your policy/model.
            # actions = policy.select_actions(obs, state, avail_actions, high_level=policy_context)

            # TODO: Step the environment and run your normal training update.
            # reward, terminated, info = env.step(actions)
            # learner.train(...)

            if step % 40 == 0:
                print(f"step={step} tactical_command={tactical_command}")
                print(f"policy_context={policy_context}")


def build_stub_obs_summary(step: int) -> dict[str, Any]:
    enemy_health = max(0.0, 45.0 - step * 0.4)
    ally_health = max(0.0, 35.0 - step * 0.1)
    return {
        "map_name": "3m",
        "allies": [
            {
                "id": "ally_0",
                "unit_type": "Marine",
                "health": ally_health,
                "shield": 0.0,
                "x": 12.0,
                "y": 8.0,
                "weapon_cooldown": 2.0,
                "is_alive": ally_health > 0,
            }
        ],
        "enemies": [
            {
                "id": "enemy_0",
                "unit_type": "Marine",
                "health": enemy_health,
                "shield": 0.0,
                "x": 14.0,
                "y": 8.5,
                "is_alive": enemy_health > 0,
            }
        ],
        "relations": [
            {
                "subject": "ally_0",
                "predicate": "nearEnemy",
                "object": "enemy_0",
                "distance": 2.1,
            }
        ],
    }


if __name__ == "__main__":
    main()

