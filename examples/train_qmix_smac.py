from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from smac_tg.qmix import QMixConfig, train_qmix


def main() -> None:
    args = parse_args()
    config = QMixConfig(
        map_name=args.map_name,
        total_timesteps=args.total_timesteps,
        seed=args.seed,
        device=args.device,
        env_args=parse_env_args(args.env_args),
        replay_size=args.replay_size,
        batch_size=args.batch_size,
        learn_start=args.learn_start,
        train_interval=args.train_interval,
        gamma=args.gamma,
        learning_rate=args.learning_rate,
        grad_norm_clip=args.grad_norm_clip,
        double_q=not args.no_double_q,
        rnn_hidden_dim=args.rnn_hidden_dim,
        mixing_embed_dim=args.mixing_embed_dim,
        hypernet_embed=args.hypernet_embed,
        target_update_interval=args.target_update_interval,
        epsilon_start=args.epsilon_start,
        epsilon_finish=args.epsilon_finish,
        epsilon_anneal_time=args.epsilon_anneal_time,
        evaluate_interval=args.evaluate_interval,
        evaluate_episodes=args.evaluate_episodes,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        checkpoint_dir=args.checkpoint_dir,
    )
    train_qmix(config)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a QMIX controller on a SMAC map.")
    parser.add_argument("--map-name", default="3m")
    parser.add_argument("--total-timesteps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--env-args",
        default="{}",
        help="JSON object forwarded to StarCraft2Env, e.g. '{\"difficulty\":\"7\"}'.",
    )

    parser.add_argument("--replay-size", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learn-start", type=int, default=2_000)
    parser.add_argument("--train-interval", type=int, default=1)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--grad-norm-clip", type=float, default=10.0)
    parser.add_argument("--no-double-q", action="store_true")

    parser.add_argument("--rnn-hidden-dim", type=int, default=64)
    parser.add_argument("--mixing-embed-dim", type=int, default=32)
    parser.add_argument("--hypernet-embed", type=int, default=64)
    parser.add_argument("--target-update-interval", type=int, default=200)

    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-finish", type=float, default=0.05)
    parser.add_argument("--epsilon-anneal-time", type=int, default=50_000)

    parser.add_argument("--evaluate-interval", type=int, default=10_000)
    parser.add_argument("--evaluate-episodes", type=int, default=8)
    parser.add_argument("--log-interval", type=int, default=2_000)
    parser.add_argument("--save-interval", type=int, default=50_000)
    parser.add_argument("--checkpoint-dir", default="runs/qmix")
    return parser.parse_args()


def parse_env_args(raw: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"--env-args must be valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise SystemExit("--env-args must decode to a JSON object")
    return parsed


if __name__ == "__main__":
    main()
