"""Evaluate a learned checkpoint against deterministic rule opponents."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from whist_ai.engine import WhistGame
from whist_ai.model import LearnedPolicy, load_checkpoint
from whist_ai.policies import RulePolicy
from whist_ai.simulator import GameController


def evaluate(checkpoint: Path, episodes: int, seed: int, device: str) -> dict[str, float]:
    network = load_checkpoint(checkpoint, device=device)
    policy = LearnedPolicy(network, device=device, deterministic=True)
    rewards = []
    for episode in range(episodes):
        controller = GameController(
            WhistGame(seed=seed + episode),
            [policy, RulePolicy(), RulePolicy(), RulePolicy()],
        )
        rewards.append(controller.run()[0])
    values = np.asarray(rewards, dtype=float)
    return {
        "episodes": float(len(values)),
        "average_reward": float(values.mean()),
        "positive_rate": float((values > 0).mean()),
        "success_rate": float((values >= 0).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20_000)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    result = evaluate(args.checkpoint, args.episodes, args.seed, args.device)
    for key, value in result.items():
        print(f"{key}={value:.4f}" if isinstance(value, float) else f"{key}={value}")


if __name__ == "__main__":
    main()
