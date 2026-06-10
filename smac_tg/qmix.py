from __future__ import annotations

import json
import random
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class QMixConfig:
    map_name: str = "3m"
    total_timesteps: int = 200_000
    seed: int = 1
    device: str = "auto"
    env_args: dict[str, Any] = field(default_factory=dict)

    replay_size: int = 5_000
    batch_size: int = 32
    learn_start: int = 2_000
    train_interval: int = 1
    gamma: float = 0.99
    learning_rate: float = 5e-4
    grad_norm_clip: float = 10.0
    double_q: bool = True

    rnn_hidden_dim: int = 64
    mixing_embed_dim: int = 32
    hypernet_embed: int = 64
    target_update_interval: int = 200

    epsilon_start: float = 1.0
    epsilon_finish: float = 0.05
    epsilon_anneal_time: int = 50_000

    evaluate_interval: int = 10_000
    evaluate_episodes: int = 8
    log_interval: int = 2_000
    save_interval: int = 50_000
    checkpoint_dir: str = "runs/qmix"


@dataclass
class EpisodeBatch:
    obs: np.ndarray
    state: np.ndarray
    avail_actions: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    terminated: np.ndarray
    filled: np.ndarray
    episode_return: float
    episode_length: int
    battle_won: bool


class EpisodeReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self._buffer: deque[EpisodeBatch] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self._buffer)

    def insert(self, episode: EpisodeBatch) -> None:
        self._buffer.append(episode)

    def can_sample(self, batch_size: int) -> bool:
        return len(self._buffer) >= batch_size

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        episodes = random.sample(list(self._buffer), batch_size)
        return {
            "obs": np.stack([episode.obs for episode in episodes]),
            "state": np.stack([episode.state for episode in episodes]),
            "avail_actions": np.stack([episode.avail_actions for episode in episodes]),
            "actions": np.stack([episode.actions for episode in episodes]),
            "rewards": np.stack([episode.rewards for episode in episodes]),
            "terminated": np.stack([episode.terminated for episode in episodes]),
            "filled": np.stack([episode.filled for episode in episodes]),
        }


class AgentRNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, n_actions: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.rnn = nn.GRUCell(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_actions)

    def init_hidden(self, batch_agents: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_agents, self.hidden_dim, device=device)

    def forward(self, inputs: torch.Tensor, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.fc1(inputs))
        next_hidden = self.rnn(x, hidden)
        q_values = self.fc2(next_hidden)
        return q_values, next_hidden


class QMixer(nn.Module):
    def __init__(self, n_agents: int, state_dim: int, embed_dim: int, hypernet_embed: int):
        super().__init__()
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.embed_dim = embed_dim

        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, hypernet_embed),
            nn.ReLU(),
            nn.Linear(hypernet_embed, n_agents * embed_dim),
        )
        self.hyper_b1 = nn.Linear(state_dim, embed_dim)

        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, hypernet_embed),
            nn.ReLU(),
            nn.Linear(hypernet_embed, embed_dim),
        )
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, agent_qs: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        batch_size, timesteps, _ = agent_qs.shape
        flat_states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.reshape(-1, 1, self.n_agents)

        w1 = torch.abs(self.hyper_w1(flat_states))
        b1 = self.hyper_b1(flat_states)
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)

        hidden = F.elu(torch.bmm(agent_qs, w1) + b1)

        w2 = torch.abs(self.hyper_w2(flat_states)).view(-1, self.embed_dim, 1)
        b2 = self.hyper_b2(flat_states).view(-1, 1, 1)

        q_tot = torch.bmm(hidden, w2) + b2
        return q_tot.view(batch_size, timesteps, 1)


class QMixLearner:
    def __init__(self, env_info: dict[str, Any], config: QMixConfig):
        self.config = config
        self.device = resolve_device(config.device)

        self.n_agents = int(env_info["n_agents"])
        self.n_actions = int(env_info["n_actions"])
        self.obs_dim = int(env_info["obs_shape"])
        self.state_dim = int(env_info["state_shape"])
        self.episode_limit = int(env_info["episode_limit"])

        self.agent_input_dim = self.obs_dim + self.n_actions + self.n_agents
        self.agent = AgentRNN(self.agent_input_dim, config.rnn_hidden_dim, self.n_actions).to(self.device)
        self.target_agent = AgentRNN(self.agent_input_dim, config.rnn_hidden_dim, self.n_actions).to(self.device)
        self.mixer = QMixer(
            self.n_agents,
            self.state_dim,
            config.mixing_embed_dim,
            config.hypernet_embed,
        ).to(self.device)
        self.target_mixer = QMixer(
            self.n_agents,
            self.state_dim,
            config.mixing_embed_dim,
            config.hypernet_embed,
        ).to(self.device)

        self.optimizer = torch.optim.RMSprop(
            list(self.agent.parameters()) + list(self.mixer.parameters()),
            lr=config.learning_rate,
            alpha=0.99,
            eps=1e-5,
        )

        self.agent_ids = torch.eye(self.n_agents, device=self.device)
        self.train_step = 0
        self.update_targets()

    def update_targets(self) -> None:
        self.target_agent.load_state_dict(self.agent.state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())

    @torch.no_grad()
    def select_actions(
        self,
        obs: np.ndarray,
        last_actions: np.ndarray,
        avail_actions: np.ndarray,
        hidden: torch.Tensor,
        epsilon: float,
        test_mode: bool = False,
    ) -> tuple[list[int], torch.Tensor]:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        last_actions_t = torch.as_tensor(last_actions, dtype=torch.float32, device=self.device)
        inputs = torch.cat([obs_t, last_actions_t, self.agent_ids], dim=-1)
        q_values, next_hidden = self.agent(inputs, hidden)

        avail_t = torch.as_tensor(avail_actions, dtype=torch.bool, device=self.device)
        q_values = q_values.masked_fill(~avail_t, -1e9)
        greedy_actions = q_values.argmax(dim=-1).cpu().numpy()

        actions: list[int] = []
        explore = (not test_mode) and epsilon > 0.0
        for agent_id, greedy_action in enumerate(greedy_actions):
            available = np.flatnonzero(avail_actions[agent_id])
            if explore and random.random() < epsilon:
                actions.append(int(np.random.choice(available)))
            else:
                actions.append(int(greedy_action))
        return actions, next_hidden

    def train(self, batch_np: dict[str, np.ndarray]) -> dict[str, float]:
        batch = {
            key: torch.as_tensor(value, device=self.device)
            for key, value in batch_np.items()
        }
        obs = batch["obs"].float()
        states = batch["state"].float()
        avail_actions = batch["avail_actions"].bool()
        actions = batch["actions"].long()
        rewards = batch["rewards"].float()
        terminated = batch["terminated"].float()
        mask = batch["filled"].float()

        mac_out = self._forward_agent_sequence(obs, actions, target=False)
        target_mac_out = self._forward_agent_sequence(obs, actions, target=True)

        chosen_action_qs = torch.gather(
            mac_out[:, :-1],
            dim=3,
            index=actions.unsqueeze(-1),
        ).squeeze(-1)

        target_next_qs = target_mac_out[:, 1:]
        target_next_qs = target_next_qs.masked_fill(~avail_actions[:, 1:], -1e9)

        if self.config.double_q:
            live_next_qs = mac_out[:, 1:].detach().masked_fill(~avail_actions[:, 1:], -1e9)
            next_actions = live_next_qs.argmax(dim=3, keepdim=True)
            target_max_qs = torch.gather(target_next_qs, dim=3, index=next_actions).squeeze(-1)
        else:
            target_max_qs = target_next_qs.max(dim=3).values

        chosen_q_tot = self.mixer(chosen_action_qs, states[:, :-1])
        target_q_tot = self.target_mixer(target_max_qs, states[:, 1:])

        targets = rewards + self.config.gamma * (1.0 - terminated) * target_q_tot.detach()
        td_error = chosen_q_tot - targets
        masked_td_error = td_error * mask
        loss = (masked_td_error.square().sum() / mask.sum().clamp(min=1.0))

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            list(self.agent.parameters()) + list(self.mixer.parameters()),
            self.config.grad_norm_clip,
        )
        self.optimizer.step()

        self.train_step += 1
        if self.train_step % self.config.target_update_interval == 0:
            self.update_targets()

        grad_norm_value = (
            float(grad_norm.detach().cpu().item())
            if isinstance(grad_norm, torch.Tensor)
            else float(grad_norm)
        )
        return {
            "loss": float(loss.detach().cpu().item()),
            "grad_norm": grad_norm_value,
            "mean_chosen_q": float(chosen_q_tot.detach().mean().cpu().item()),
            "mean_target_q": float(target_q_tot.detach().mean().cpu().item()),
        }

    def _forward_agent_sequence(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        target: bool,
    ) -> torch.Tensor:
        batch_size = obs.shape[0]
        agent = self.target_agent if target else self.agent
        hidden = agent.init_hidden(batch_size * self.n_agents, self.device)
        outputs: list[torch.Tensor] = []

        zero_last_actions = torch.zeros(
            batch_size,
            self.n_agents,
            self.n_actions,
            device=self.device,
            dtype=torch.float32,
        )
        action_onehot = F.one_hot(actions, num_classes=self.n_actions).float()

        for timestep in range(self.episode_limit + 1):
            if timestep == 0:
                last_actions = zero_last_actions
            else:
                last_actions = action_onehot[:, timestep - 1]

            agent_ids = self.agent_ids.unsqueeze(0).expand(batch_size, -1, -1)
            inputs = torch.cat([obs[:, timestep], last_actions, agent_ids], dim=-1)
            q_values, hidden = agent(
                inputs.reshape(batch_size * self.n_agents, -1),
                hidden,
            )
            outputs.append(q_values.view(batch_size, self.n_agents, self.n_actions))

        return torch.stack(outputs, dim=1)

    def save_checkpoint(self, path: str | Path, env_steps: int) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "agent": self.agent.state_dict(),
                "target_agent": self.target_agent.state_dict(),
                "mixer": self.mixer.state_dict(),
                "target_mixer": self.target_mixer.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "train_step": self.train_step,
                "env_steps": env_steps,
                "config": asdict(self.config),
            },
            path,
        )

    def load_checkpoint(self, path: str | Path) -> int:
        checkpoint = torch.load(path, map_location=self.device)
        self.agent.load_state_dict(checkpoint["agent"])
        self.target_agent.load_state_dict(checkpoint.get("target_agent", checkpoint["agent"]))
        self.mixer.load_state_dict(checkpoint["mixer"])
        self.target_mixer.load_state_dict(checkpoint.get("target_mixer", checkpoint["mixer"]))
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.train_step = int(checkpoint.get("train_step", 0))
        return int(checkpoint.get("env_steps", 0))


class SMACEpisodeRunner:
    def __init__(self, env: Any, learner: QMixLearner):
        self.env = env
        self.learner = learner

    def run(self, epsilon: float, test_mode: bool = False) -> EpisodeBatch:
        episode_limit = self.learner.episode_limit
        n_agents = self.learner.n_agents
        n_actions = self.learner.n_actions
        obs_dim = self.learner.obs_dim
        state_dim = self.learner.state_dim

        obs_buffer = np.zeros((episode_limit + 1, n_agents, obs_dim), dtype=np.float32)
        state_buffer = np.zeros((episode_limit + 1, state_dim), dtype=np.float32)
        avail_buffer = np.zeros((episode_limit + 1, n_agents, n_actions), dtype=np.float32)
        actions_buffer = np.zeros((episode_limit, n_agents), dtype=np.int64)
        rewards_buffer = np.zeros((episode_limit, 1), dtype=np.float32)
        terminated_buffer = np.zeros((episode_limit, 1), dtype=np.float32)
        filled_buffer = np.zeros((episode_limit, 1), dtype=np.float32)

        self.env.reset()
        obs_buffer[0] = np.asarray(self.env.get_obs(), dtype=np.float32)
        state_buffer[0] = np.asarray(self.env.get_state(), dtype=np.float32)
        avail_buffer[0] = np.asarray(self.env.get_avail_actions(), dtype=np.float32)

        hidden = self.learner.agent.init_hidden(n_agents, self.learner.device)
        last_actions = np.zeros((n_agents, n_actions), dtype=np.float32)
        episode_return = 0.0
        episode_length = 0
        battle_won = False

        for timestep in range(episode_limit):
            actions, hidden = self.learner.select_actions(
                obs=obs_buffer[timestep],
                last_actions=last_actions,
                avail_actions=avail_buffer[timestep],
                hidden=hidden,
                epsilon=epsilon,
                test_mode=test_mode,
            )
            reward, terminated, info = self.env.step(actions)
            battle_won = bool(info.get("battle_won", battle_won)) if isinstance(info, dict) else battle_won

            actions_buffer[timestep] = np.asarray(actions, dtype=np.int64)
            rewards_buffer[timestep, 0] = float(reward)
            terminated_buffer[timestep, 0] = float(terminated)
            filled_buffer[timestep, 0] = 1.0

            last_actions = np.zeros((n_agents, n_actions), dtype=np.float32)
            last_actions[np.arange(n_agents), actions] = 1.0
            episode_return += float(reward)
            episode_length = timestep + 1

            if terminated:
                break

            obs_buffer[timestep + 1] = np.asarray(self.env.get_obs(), dtype=np.float32)
            state_buffer[timestep + 1] = np.asarray(self.env.get_state(), dtype=np.float32)
            avail_buffer[timestep + 1] = np.asarray(self.env.get_avail_actions(), dtype=np.float32)

        return EpisodeBatch(
            obs=obs_buffer,
            state=state_buffer,
            avail_actions=avail_buffer,
            actions=actions_buffer,
            rewards=rewards_buffer,
            terminated=terminated_buffer,
            filled=filled_buffer,
            episode_return=episode_return,
            episode_length=episode_length,
            battle_won=battle_won,
        )


def train_qmix(config: QMixConfig) -> None:
    set_global_seeds(config.seed)
    env = make_smac_env(config)
    eval_env = make_smac_env(config)

    try:
        env_info = env.get_env_info()
        learner = QMixLearner(env_info, config)
        runner = SMACEpisodeRunner(env, learner)
        eval_runner = SMACEpisodeRunner(eval_env, learner)
        replay = EpisodeReplayBuffer(config.replay_size)

        checkpoint_root = Path(config.checkpoint_dir) / config.map_name / time.strftime("%Y%m%d-%H%M%S")
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        (checkpoint_root / "config.json").write_text(
            json.dumps(asdict(config), indent=2, sort_keys=True),
            encoding="utf-8",
        )

        env_steps = 0
        next_log = config.log_interval
        next_eval = config.evaluate_interval
        next_save = config.save_interval
        recent_returns: deque[float] = deque(maxlen=100)
        recent_wins: deque[float] = deque(maxlen=100)
        last_train_stats: dict[str, float] = {}
        episode_idx = 0

        while env_steps < config.total_timesteps:
            epsilon = epsilon_by_step(config, env_steps)
            episode = runner.run(epsilon=epsilon, test_mode=False)
            replay.insert(episode)
            env_steps += episode.episode_length
            episode_idx += 1
            recent_returns.append(episode.episode_return)
            recent_wins.append(float(episode.battle_won))

            if replay.can_sample(config.batch_size) and env_steps >= config.learn_start:
                for _ in range(config.train_interval):
                    last_train_stats = learner.train(replay.sample(config.batch_size))

            if env_steps >= next_log:
                log_training_status(
                    env_steps=env_steps,
                    episode_idx=episode_idx,
                    epsilon=epsilon,
                    returns=recent_returns,
                    wins=recent_wins,
                    replay_size=len(replay),
                    train_stats=last_train_stats,
                )
                next_log += config.log_interval

            if env_steps >= next_eval:
                evaluate(eval_runner, config, env_steps)
                next_eval += config.evaluate_interval

            if env_steps >= next_save:
                checkpoint_path = checkpoint_root / f"step_{env_steps}.pt"
                learner.save_checkpoint(checkpoint_path, env_steps=env_steps)
                print(f"[checkpoint] saved {checkpoint_path}")
                next_save += config.save_interval

        final_path = checkpoint_root / "final.pt"
        learner.save_checkpoint(final_path, env_steps=env_steps)
        print(f"[done] env_steps={env_steps} checkpoint={final_path}")
    finally:
        env.close()
        eval_env.close()


def evaluate(runner: SMACEpisodeRunner, config: QMixConfig, env_steps: int) -> None:
    returns = []
    wins = []
    for _ in range(config.evaluate_episodes):
        episode = runner.run(epsilon=0.0, test_mode=True)
        returns.append(episode.episode_return)
        wins.append(float(episode.battle_won))
    print(
        "[eval] "
        f"env_steps={env_steps} "
        f"mean_return={np.mean(returns):.3f} "
        f"win_rate={np.mean(wins):.3f}"
    )


def log_training_status(
    env_steps: int,
    episode_idx: int,
    epsilon: float,
    returns: deque[float],
    wins: deque[float],
    replay_size: int,
    train_stats: dict[str, float],
) -> None:
    stats = " ".join(f"{key}={value:.4f}" for key, value in train_stats.items())
    print(
        "[train] "
        f"env_steps={env_steps} "
        f"episodes={episode_idx} "
        f"epsilon={epsilon:.3f} "
        f"mean_return_100={np.mean(returns):.3f} "
        f"win_rate_100={np.mean(wins):.3f} "
        f"replay={replay_size} "
        f"{stats}"
    )


def epsilon_by_step(config: QMixConfig, env_steps: int) -> float:
    if config.epsilon_anneal_time <= 0:
        return config.epsilon_finish
    progress = min(1.0, env_steps / config.epsilon_anneal_time)
    return config.epsilon_start + progress * (config.epsilon_finish - config.epsilon_start)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_smac_env(config: QMixConfig) -> Any:
    try:
        from smac.env import StarCraft2Env
    except ImportError as exc:
        raise RuntimeError(
            "Cannot import SMAC. Install the OxWhirl SMAC package and configure "
            "StarCraft II before running QMIX training."
        ) from exc

    env = StarCraft2Env(map_name=config.map_name, **config.env_args)
    if hasattr(env, "seed"):
        env.seed(config.seed)
    return env
