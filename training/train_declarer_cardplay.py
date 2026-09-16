"""Train declarer card play against fixed rule-controlled teammates and opponents."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from training.train import update_policy
from training.plot_metrics import plot_declarer_cardplay
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


POLICY_ROLE = "declarer_card_play"
DEFAULT_CONTRACTS = (7, 8, 9)


def parse_contracts(value: str) -> tuple[int, ...]:
    contracts = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not contracts or any(contract not in range(7, 14) for contract in contracts):
        raise ValueError("contracts must be a comma-separated list of integers from 7 through 13")
    return contracts


def train(
    episodes: int,
    seed: int,
    contracts: tuple[int, ...],
    checkpoint: Path,
    device: str,
    update_epochs: int,
    memory_limit: float,
    memory_resume: float,
    gpu_temperature_limit: float,
    gpu_temperature_resume: float,
    gpu_memory_limit: float,
    metrics_path: Path,
    graph_path: Path,
    graph_every: int,
) -> None:
    torch.manual_seed(seed)
    checkpoint_data = None
    if checkpoint.exists():
        network, checkpoint_data = load_training_checkpoint(checkpoint, device=device)
        if checkpoint_data.get("policy_role") != POLICY_ROLE:
            raise ValueError(
                f"checkpoint must have policy_role={POLICY_ROLE!r}; start this curriculum with a fresh checkpoint"
            )
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
                    "episode", "seed", "contract", "declarer", "learner_role",
                    "hand_strength", "reward", "declarer_tricks", "team_tricks",
                    "success", "decisions", "loss",
                )
            )
        for episode in range(start_episode + 1, start_episode + episodes + 1):
            wait_for_memory(memory_limit, memory_resume)
            if device.startswith("cuda"):
                wait_for_gpu(gpu_temperature_limit, gpu_temperature_resume, gpu_memory_limit)

            game_seed = seed + episode
            game = WhistGame(seed=game_seed, dealer=game_seed % 4)
            declarer = (episode - 1) % 4
            contract = contracts[(episode - 1) % len(contracts)]
            hand = tuple(game.hands[declarer])
            trump_suit = game_seed % 4
            partner_suit = (trump_suit + 1) % 4
            game.start_fixed_contract(
                contract,
                declarer=declarer,
                trump_suit=trump_suit,
                partner_suit=partner_suit,
            )

            opponents = [
                RulePolicy(("conservative", "balanced", "aggressive")[seat % 3])
                for seat in range(4)
            ]
            opponents[declarer] = policy
            controller = GameController(game, opponents)
            rewards = controller.run()
            reward = float(rewards[declarer])
            loss = update_policy(policy, optimizer, reward, update_epochs)
            team_tricks = game.tricks_won[declarer] + game.tricks_won[game.partner_player]
            writer.writerow(
                (
                    episode,
                    game_seed,
                    contract,
                    declarer,
                    "declarer",
                    sum(max(0, rank - 8) for _, rank in hand),
                    reward,
                    game.tricks_won[declarer],
                    team_tricks,
                    int(reward >= 0),
                    len(controller.decisions),
                    loss,
                )
            )
            metrics_file.flush()
            if episode == start_episode + 1 or episode % 100 == 0:
                print(
                    f"episode={episode} contract={contract} declarer={declarer} "
                    f"reward={reward:+.1f} team_tricks={team_tricks} loss={loss:.4f}",
                    flush=True,
                )
            if episode % 1000 == 0:
                save_checkpoint(network, checkpoint, optimizer, episode, policy_role=POLICY_ROLE)
            if episode % graph_every == 0:
                plot_declarer_cardplay(metrics_path, graph_path, window=100)

    save_checkpoint(network, checkpoint, optimizer, start_episode + episodes, policy_role=POLICY_ROLE)
    plot_declarer_cardplay(metrics_path, graph_path, window=100)
    print(f"saved={checkpoint} metrics={metrics_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=150_000)
    parser.add_argument("--contracts", type=parse_contracts, default=DEFAULT_CONTRACTS)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/declarer_cardplay_7_9.pt"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--memory-limit", type=float, default=80.0)
    parser.add_argument("--memory-resume", type=float, default=75.0)
    parser.add_argument("--gpu-temperature-limit", type=float, default=80.0)
    parser.add_argument("--gpu-temperature-resume", type=float, default=70.0)
    parser.add_argument("--gpu-memory-limit", type=float, default=90.0)
    parser.add_argument("--metrics", type=Path, default=Path("graphs/declarer_cardplay_7_9_metrics.csv"))
    parser.add_argument("--graph", type=Path, default=Path("graphs/declarer_cardplay_progress.png"))
    parser.add_argument("--graph-every", type=int, default=1000)
    args = parser.parse_args()
    if args.episodes <= 0 or args.update_epochs <= 0 or args.graph_every <= 0:
        raise ValueError("episodes, update epochs, and graph frequency must be positive")
    train(
        args.episodes,
        args.seed,
        args.contracts,
        args.checkpoint,
        args.device,
        args.update_epochs,
        args.memory_limit,
        args.memory_resume,
        args.gpu_temperature_limit,
        args.gpu_temperature_resume,
        args.gpu_memory_limit,
        args.metrics,
        args.graph,
        args.graph_every,
    )


if __name__ == "__main__":
    main()
