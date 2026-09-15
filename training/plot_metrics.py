"""Plot training progress and the project's terminal reward curves."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

from whist_ai.rewards import NUMERIC_CONTRACTS, NOLO_REWARDS, nolo_reward, numeric_reward


def read_metrics(path: Path) -> dict[str, list[float]]:
    with path.open(newline="", encoding="utf-8") as metrics_file:
        rows = list(csv.DictReader(metrics_file))
    if not rows:
        raise ValueError(f"no training rows found in {path}")
    return {
        key: [float(row[key]) for row in rows]
        for key in ("episode", "reward", "loss", "positive", "nonnegative")
    }


def plot_progress(metrics_path: Path, output_path: Path, window: int) -> None:
    metrics = read_metrics(metrics_path)
    rewards = metrics["reward"]
    rolling = [
        sum(rewards[max(0, index - window + 1) : index + 1])
        / len(rewards[max(0, index - window + 1) : index + 1])
        for index in range(len(rewards))
    ]
    figure, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(metrics["episode"], rewards, alpha=0.25, label="episode reward")
    axes[0].plot(metrics["episode"], rolling, label=f"{window}-episode average")
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_ylabel("game points")
    axes[0].legend()
    axes[1].plot(metrics["episode"], metrics["loss"], label="loss")
    axes[1].plot(metrics["episode"], metrics["positive"], label="positive reward")
    axes[1].plot(metrics["episode"], metrics["nonnegative"], label="nonnegative reward")
    axes[1].set_xlabel("episode")
    axes[1].legend()
    figure.suptitle("Whist policy training progress")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def plot_rewards(output_path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    tricks = list(range(14))
    for bid in NUMERIC_CONTRACTS:
        axes[0].plot(tricks, [numeric_reward(bid, count) for count in tricks], label=f"bid {bid}")
    axes[0].set_title("Numeric contract reward")
    axes[0].set_xlabel("tricks taken")
    axes[0].set_ylabel("game points")
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].legend(ncol=2, fontsize="small")
    for contract, (_, success, failure) in NOLO_REWARDS.items():
        values = [nolo_reward(contract, count) for count in tricks]
        axes[1].plot(tricks, values, label=contract)
        axes[1].axhline(success, linestyle="--", linewidth=0.7)
        axes[1].axhline(failure, linestyle=":", linewidth=0.7)
    axes[1].set_title("Nolo contract reward")
    axes[1].set_xlabel("tricks taken")
    axes[1].set_ylabel("game points")
    axes[1].legend()
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def plot_bidding(decisions_path: Path, output_path: Path, window: int) -> None:
    with decisions_path.open(newline="", encoding="utf-8") as decisions_file:
        rows = list(csv.DictReader(decisions_file))
    if not rows:
        raise ValueError(f"no bidding rows found in {decisions_path}")
    contracts = sorted({f"{row['contract']} (P{row['declarer']})" for row in rows})
    episodes = [int(row["episode"]) for row in rows]
    figure, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    for contract in contracts:
        counts = [sum(f"{row['contract']} (P{row['declarer']})" == contract for row in rows[: index + 1])
                  for index in range(len(rows))]
        axes[0].plot(episodes, counts, label=contract)
    axes[0].set_ylabel("cumulative contracts")
    axes[0].set_title("Contracts selected during training")
    axes[0].legend(ncol=3, fontsize="small")
    for player in range(4):
        bid_counts = []
        for index in range(len(rows)):
            recent = rows[max(0, index - window + 1) : index + 1]
            bid_counts.append(
                sum(
                    1
                    for recent_row in recent
                    for decision in json.loads(recent_row["bid_history"])
                    if decision["player"] == player and decision["action"] != "pass"
                )
            )
        axes[1].plot(episodes, bid_counts, label=f"player {player}")
    axes[1].set_xlabel("episode")
    axes[1].set_ylabel(f"bids in last {window} games")
    axes[1].set_title("Non-pass bids by player")
    axes[1].legend()
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=Path("graphs/training_metrics.csv"))
    parser.add_argument("--decisions", type=Path, default=Path("graphs/bidding_decisions.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("graphs"))
    parser.add_argument("--window", type=int, default=100)
    args = parser.parse_args()
    if args.window <= 0:
        raise ValueError("window must be positive")
    plot_progress(args.metrics, args.output_dir / "training_progress.png", args.window)
    plot_rewards(args.output_dir / "reward_structure.png")
    plot_bidding(args.decisions, args.output_dir / "bidding_progress.png", args.window)
    print(f"saved={args.output_dir / 'training_progress.png'}")
    print(f"saved={args.output_dir / 'reward_structure.png'}")
    print(f"saved={args.output_dir / 'bidding_progress.png'}")


if __name__ == "__main__":
    main()