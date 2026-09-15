"""Plot training progress and the project's terminal reward curves."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
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
    episodes = metrics["episode"]
    rewards = metrics["reward"]
    # Episode numbers restart when a resumed run writes to a new or shared CSV.
    # Recording order remains monotonic and is the honest x-axis for mixed runs.
    recorded_games = list(range(1, len(rewards) + 1))
    bins = list(range(0, len(rewards), window))
    mean_rewards = [sum(rewards[start : start + window]) / len(rewards[start : start + window]) for start in bins]
    positive_rates = [
        sum(metrics["positive"][start : start + window]) / len(metrics["positive"][start : start + window])
        for start in bins
    ]
    losses = [sum(metrics["loss"][start : start + window]) / len(metrics["loss"][start : start + window]) for start in bins]
    labels = [recorded_games[start] for start in bins]
    figure, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    axes[0].plot(labels, mean_rewards, color="#1769aa", marker="o", markersize=2, label="average reward")
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_ylabel("game points")
    axes[0].set_title(f"Average reward per {window}-episode block")
    axes[0].legend()
    axes[1].plot(labels, positive_rates, color="#2e7d32", label="positive reward rate")
    axes[1].plot(labels, [1 - value for value in positive_rates], color="#c62828", label="negative reward rate")
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("share of games")
    axes[1].set_title("Positive versus negative game outcomes")
    axes[1].legend()
    axes[2].plot(labels, losses, color="#6a1b9a", label="average training loss")
    axes[2].set_xlabel("games recorded")
    axes[2].set_ylabel("loss")
    axes[2].set_title("Training loss")
    axes[2].legend()
    figure.suptitle("Whist policy training dashboard")
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
    contract_order = ["paskrig", "sol", "ren sol", "bordlaegger"] + [str(value) for value in range(7, 14)]
    bid_order = [str(value) for value in range(7, 14)] + ["sol", "ren sol", "bordlaegger"]
    contract_counts = {player: Counter() for player in range(4)}
    bid_counts = {player: Counter() for player in range(4)}
    for row in rows:
        declarer = row["declarer"]
        if declarer != "":
            contract_counts[int(declarer)][row["contract"]] += 1
        for decision in json.loads(row["bid_history"]):
            if decision["action"] != "pass":
                bid_counts[int(decision["player"])][str(decision["action"])] += 1
    figure, axes = plt.subplots(2, 1, figsize=(12, 9))
    player_labels = ["P0 learned", "P1", "P2", "P3"]
    for axis, counts, labels, title in (
        (axes[0], contract_counts, contract_order, "Contracts won by each declarer"),
        (axes[1], bid_counts, bid_order, "Non-pass bids made by each player"),
    ):
        matrix = [[counts[player][label] for label in labels] for player in range(4)]
        totals = [sum(row) for row in matrix]
        percentages = [
            [100 * value / total if total else 0 for value in row]
            for row, total in zip(matrix, totals)
        ]
        image = axis.imshow(percentages, aspect="auto", cmap="YlGnBu", vmin=0, vmax=100)
        axis.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
        axis.set_yticks(range(4), [f"{label} (n={total})" for label, total in zip(player_labels, totals)])
        axis.set_title(title)
        axis.set_xlabel("contract" if axis is axes[0] else "bid")
        axis.set_ylabel("player")
        for player in range(4):
            for index, value in enumerate(percentages[player]):
                if value:
                    axis.text(index, player, f"{value:.0f}%", ha="center", va="center", fontsize=8)
        figure.colorbar(image, ax=axis, label="share of this player's actions (%)")
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