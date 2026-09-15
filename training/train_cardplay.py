"""Train the card-play specialist from fixed-contract scenarios."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from whist_ai.engine import WhistGame
from whist_ai.model import (
    LearnedPolicy,
    PolicyNetwork,
    load_training_checkpoint,
    save_checkpoint,
)
from whist_ai.policies import RulePolicy
from whist_ai.resources import wait_for_gpu, wait_for_memory
from whist_ai.simulator import GameController

from training.train import update_policy


CONTRACTS = (7, 8, 9, 10, 11, 12, 13, "sol", "ren sol", "bordlaegger", "paskrig")


def hand_strength(hand: tuple[tuple[int, int], ...]) -> int:
    points = sum(max(0, rank - 8) for _, rank in hand)
    longest_suit = max(sum(card[0] == suit for card in hand) for suit in range(4))
    return points + max(0, longest_suit - 4)


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
    checkpoint_data = None
    if checkpoint.exists():
        network, checkpoint_data = load_training_checkpoint(checkpoint, device=device)
        if checkpoint_data.get("policy_role") not in {"card_play", "shared_full_game"}:
            raise ValueError("checkpoint is not compatible with card-play training")
        print(f"resuming={checkpoint}")
    else:
        network = PolicyNetwork().to(device)
    policy = LearnedPolicy(network, device=device)
    optimizer = torch.optim.Adam(network.parameters(), lr=3e-4)
    start_episode = 0
    if checkpoint_data is not None:
        start_episode = int(checkpoint_data.get("episode", 0))
        if "optimizer" in checkpoint_data:
            optimizer.load_state_dict(checkpoint_data["optimizer"])

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    exists = metrics_path.exists()
    with metrics_path.open("a", newline="", encoding="utf-8") as metrics_file:
        writer = csv.writer(metrics_file)
        if not exists:
            writer.writerow(
                (
                    "episode", "seed", "contract", "learner_seat", "learner_role",
                    "hand_strength", "reward", "tricks", "team_tricks", "success",
                    "decisions", "loss",
                )
            )
        for episode in range(start_episode + 1, start_episode + episodes + 1):
            wait_for_memory(memory_limit, memory_resume)
            if device.startswith("cuda"):
                wait_for_gpu(gpu_temperature_limit, gpu_temperature_resume, gpu_memory_limit)
            game_seed = seed + episode
            game = WhistGame(seed=game_seed, dealer=game_seed % 4)
            learner_seat = (episode - 1) % 4
            learner_hand_strength = hand_strength(tuple(game.hands[learner_seat]))
            contract = CONTRACTS[(episode - 1) % len(CONTRACTS)]
            declarer = (game_seed + 1) % 4
            trump_suit = game_seed % 4
            partner_suit = (trump_suit + 1) % 4
            game.start_fixed_contract(
                contract,
                declarer=declarer,
                trump_suit=None if not isinstance(contract, int) else trump_suit,
                partner_suit=None if not isinstance(contract, int) else partner_suit,
            )
            opponents = [
                RulePolicy(("conservative", "balanced", "aggressive")[seat % 3])
                for seat in range(4)
            ]
            opponents[learner_seat] = policy
            controller = GameController(game, opponents)
            rewards = controller.run()
            reward = float(rewards[learner_seat])
            loss = update_policy(policy, optimizer, reward, update_epochs)
            role = "declarer" if learner_seat == declarer else (
                "partner" if learner_seat == game.partner_player else "defender"
            )
            tricks = game.tricks_won[learner_seat]
            team_tricks = (
                game.tricks_won[declarer] + game.tricks_won[game.partner_player]
                if game.partner_player is not None else tricks
            )
            writer.writerow(
                (
                    episode, game_seed, contract, learner_seat, role,
                    learner_hand_strength,
                    reward, tricks, team_tricks, int(reward >= 0),
                    len(controller.decisions), loss,
                )
            )
            metrics_file.flush()
            if episode == start_episode + 1 or episode % 100 == 0:
                print(f"episode={episode} contract={contract} role={role} reward={reward:+.1f} loss={loss:.4f}", flush=True)
            if episode % 1000 == 0:
                save_checkpoint(network, checkpoint, optimizer, episode, policy_role="card_play")
    save_checkpoint(network, checkpoint, optimizer, start_episode + episodes, policy_role="card_play")
    print(f"saved={checkpoint} metrics={metrics_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=50_000)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/cardplay_policy.pt"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--memory-limit", type=float, default=80.0)
    parser.add_argument("--memory-resume", type=float, default=75.0)
    parser.add_argument("--gpu-temperature-limit", type=float, default=80.0)
    parser.add_argument("--gpu-temperature-resume", type=float, default=70.0)
    parser.add_argument("--gpu-memory-limit", type=float, default=90.0)
    parser.add_argument("--metrics", type=Path, default=Path("graphs/cardplay_training_metrics.csv"))
    args = parser.parse_args()
    if args.episodes <= 0 or args.update_epochs <= 0:
        raise ValueError("episodes and update epochs must be positive")
    train(
        args.episodes, args.seed, args.checkpoint, args.device, args.update_epochs,
        args.memory_limit, args.memory_resume, args.gpu_temperature_limit,
        args.gpu_temperature_resume, args.gpu_memory_limit, args.metrics,
    )


if __name__ == "__main__":
    main()