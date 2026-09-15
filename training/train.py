"""Train the shared masked policy against deterministic rule opponents."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch.distributions import Categorical

from whist_ai.engine import WhistGame
from whist_ai.model import LearnedPolicy, PolicyNetwork, save_checkpoint
from whist_ai.policies import RulePolicy
from whist_ai.resources import wait_for_gpu, wait_for_memory
from whist_ai.simulator import GameController


def update_policy(policy: LearnedPolicy, optimizer: torch.optim.Optimizer, reward: float, epochs: int) -> float:
    if not policy.transitions:
        return 0.0
    losses = []
    target = torch.tensor(reward, dtype=torch.float32, device=policy.device)
    for _ in range(epochs):
        loss = torch.zeros((), device=policy.device)
        for state, phase, legal_indices, action, old_log_probability, _ in policy.transitions:
            logits, value = policy.network(state, phase)
            mask = torch.full_like(logits, float("-inf"))
            mask[list(legal_indices)] = 0.0
            distribution = Categorical(logits=logits + mask)
            log_probability = distribution.log_prob(action)
            ratio = torch.exp(log_probability - old_log_probability.detach())
            advantage = target - value.detach()
            clipped = torch.clamp(ratio, 0.8, 1.2) * advantage
            policy_loss = -torch.minimum(ratio * advantage, clipped)
            value_loss = 0.5 * (value - target).pow(2)
            loss = loss + policy_loss + value_loss
        loss = loss / len(policy.transitions)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.network.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    policy.clear_transitions()
    return sum(losses) / len(losses)


def train(
    episodes: int,
    seed: int,
    checkpoint: Path,
    device: str,
    update_epochs: int,
    memory_limit: float,
    memory_resume: float,
    gpu_temperature_limit: float,
    gpu_temperature_resume: float,
    gpu_memory_limit: float,
    metrics_path: Path,
) -> None:
    torch.manual_seed(seed)
    if checkpoint.exists():
        from whist_ai.model import load_checkpoint
        network = load_checkpoint(checkpoint, device=device)
        print(f"resuming={checkpoint}")
    else:
        network = PolicyNetwork().to(device)
    policy = LearnedPolicy(network, device=device)
    optimizer = torch.optim.Adam(network.parameters(), lr=3e-4)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_exists = metrics_path.exists()
    with metrics_path.open("a", newline="", encoding="utf-8") as metrics_file:
        metrics = csv.writer(metrics_file)
        if not metrics_exists:
            metrics.writerow(("episode", "reward", "loss", "positive", "nonnegative"))
    for episode in range(1, episodes + 1):
        wait_for_memory(memory_limit, memory_resume)
        if device.startswith("cuda"):
            wait_for_gpu(gpu_temperature_limit, gpu_temperature_resume, gpu_memory_limit)
        game = WhistGame(seed=seed + episode)
        controller = GameController(game, [policy, RulePolicy(), RulePolicy(), RulePolicy()])
        rewards = controller.run()
        reward = float(rewards[0])
        loss = update_policy(policy, optimizer, reward, update_epochs)
        with metrics_path.open("a", newline="", encoding="utf-8") as metrics_file:
            csv.writer(metrics_file).writerow(
                (episode, reward, loss, int(reward > 0), int(reward >= 0))
            )
        if episode == 1 or episode % 100 == 0:
            print(f"episode={episode} reward={reward:+.1f} loss={loss:.4f}", flush=True)
        if episode % 1000 == 0:
            save_checkpoint(network, checkpoint)
    save_checkpoint(network, checkpoint)
    print(f"saved={checkpoint} metrics={metrics_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/whist_policy.pt"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--memory-limit", type=float, default=80.0)
    parser.add_argument("--memory-resume", type=float, default=75.0)
    parser.add_argument("--gpu-temperature-limit", type=float, default=80.0)
    parser.add_argument("--gpu-temperature-resume", type=float, default=70.0)
    parser.add_argument("--gpu-memory-limit", type=float, default=90.0)
    parser.add_argument("--metrics", type=Path, default=Path("graphs/training_metrics.csv"))
    args = parser.parse_args()
    if args.episodes <= 0 or args.update_epochs <= 0:
        raise ValueError("episodes and update epochs must be positive")
    train(
        args.episodes,
        args.seed,
        args.checkpoint,
        args.device,
        args.update_epochs,
        args.memory_limit,
        args.memory_resume,
        args.gpu_temperature_limit,
        args.gpu_temperature_resume,
        args.gpu_memory_limit,
        args.metrics,
    )


if __name__ == "__main__":
    main()
