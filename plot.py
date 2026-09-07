"""Generate training and benchmark graphics."""

import csv
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REWARDS_CSV = "rewards.csv"
GRAPH_DIR = "graphs"
BENCHMARK_CSV = "benchmarks.csv"
BENCHMARK_DETAILS_CSV = "benchmark_games.csv"


def _wilson_interval(wins, games, z=1.96):
    if games <= 0:
        return 0.0, 0.0
    p = wins / games
    denominator = 1 + z * z / games
    centre = (p + z * z / (2 * games)) / denominator
    spread = z * np.sqrt((p * (1 - p) + z * z / (4 * games)) / games) / denominator
    return max(0.0, centre - spread), min(1.0, centre + spread)


def plot_dashboard(out_dir=GRAPH_DIR):
    """Create a four-panel dashboard from all available training metrics."""
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    reward_ax, win_ax, benchmark_ax, tricks_ax = axes.flat
    if os.path.exists(REWARDS_CSV):
        rewards = pd.read_csv(REWARDS_CSV).drop_duplicates("episode", keep="last")
        reward_ax.plot(rewards["episode"], rewards["avg_reward"], color="steelblue", alpha=0.45)
        if len(rewards) >= 20:
            reward_ax.plot(
                rewards["episode"], rewards["avg_reward"].rolling(20, min_periods=1).mean(),
                color="darkred", linewidth=2, label="20-point mean",
            )
        reward_ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    reward_ax.set_title("Complete Episode Return")
    reward_ax.set_xlabel("Episode")
    reward_ax.set_ylabel("Return")
    reward_ax.grid(alpha=0.25)

    if os.path.exists("winrate.csv"):
        win_rates = pd.read_csv("winrate.csv").drop_duplicates("episode", keep="last")
        win_ax.plot(win_rates["episode"], win_rates["win_rate_vs_baseline"], marker="o")
        win_ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    win_ax.set_title("Win Rate vs Frozen Baseline")
    win_ax.set_ylim(0, 1)
    win_ax.set_xlabel("Episode")
    win_ax.set_ylabel("Win rate")
    win_ax.grid(alpha=0.25)

    if os.path.exists(BENCHMARK_CSV):
        benchmarks = pd.read_csv(BENCHMARK_CSV)
        for opponent, group in benchmarks.groupby("opponent"):
            group = group.sort_values("checkpoint_episode")
            lower, upper = [], []
            for _, row in group.iterrows():
                lo, hi = _wilson_interval(int(row["wins"]), int(row["games"]))
                lower.append(lo)
                upper.append(hi)
            x = group["checkpoint_episode"]
            y = group["win_rate"]
            benchmark_ax.plot(x, y, marker="o", label=opponent)
            benchmark_ax.fill_between(x, lower, upper, alpha=0.15)
        benchmark_ax.legend()
    benchmark_ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    benchmark_ax.set_ylim(0, 1)
    benchmark_ax.set_title("Fixed Opponent Benchmarks")
    benchmark_ax.set_xlabel("Checkpoint episode")
    benchmark_ax.set_ylabel("Win rate")
    benchmark_ax.grid(alpha=0.25)

    if os.path.exists(BENCHMARK_CSV):
        benchmarks = pd.read_csv(BENCHMARK_CSV)
        for opponent, group in benchmarks.groupby("opponent"):
            group = group.sort_values("checkpoint_episode")
            tricks_ax.plot(
                group["checkpoint_episode"], group["avg_trick_difference"],
                marker="o", label=opponent,
            )
        tricks_ax.axhline(0, color="gray", linestyle="--", linewidth=1)
        tricks_ax.legend()
    tricks_ax.set_title("Average Trick Difference")
    tricks_ax.set_xlabel("Checkpoint episode")
    tricks_ax.set_ylabel("Agent tricks - opponent tricks")
    tricks_ax.grid(alpha=0.25)

    fig.suptitle("Whist Agent Training Dashboard", fontsize=16)
    fig.tight_layout()
    path = os.path.join(out_dir, "dashboard.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"Dashboard saved to {path}")


def plot_breakdowns(out_dir=GRAPH_DIR):
    """Plot benchmark win rates by agent team and trump condition."""
    if not os.path.exists(BENCHMARK_DETAILS_CSV):
        return
    details = pd.read_csv(BENCHMARK_DETAILS_CSV)
    if details.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    seat = details.groupby(["opponent", "agent_team"], as_index=False)["win"].mean()
    for opponent, group in seat.groupby("opponent"):
        axes[0].plot(group["agent_team"], group["win"], marker="o", label=opponent)
    axes[0].set_xticks([0, 1], ["Team 0", "Team 1"])
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Win Rate by Agent Team")
    axes[0].set_ylabel("Win rate")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    trump = details.copy()
    trump["trump_label"] = trump["trump_suit"].map({0: "Clubs", 1: "Diamonds", 2: "Hearts", 3: "Spades", 4: "No trump"})
    grouped = trump.groupby(["opponent", "trump_label"], as_index=False)["win"].mean()
    order = ["Clubs", "Diamonds", "Hearts", "Spades", "No trump"]
    positions = np.arange(len(order))
    for opponent, group in grouped.groupby("opponent"):
        values = group.set_index("trump_label")["win"].reindex(order)
        axes[1].plot(positions, values, marker="o", label=opponent)
    axes[1].set_xticks(positions, order, rotation=25)
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Win Rate by Trump Condition")
    axes[1].set_ylabel("Win rate")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    path = os.path.join(out_dir, "benchmark_breakdowns.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"Breakdown graph saved to {path}")


def plot_rewards(csv_path=REWARDS_CSV, out_dir=GRAPH_DIR):
    """Read the full rewards.csv and save a single continuous reward plot."""
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    if df.empty or "episode" not in df.columns or "avg_reward" not in df.columns:
        print("No data found in rewards.csv.")
        sys.exit(1)

    # Sort by episode and drop duplicate episode rows (keep last occurrence)
    df = df.sort_values("episode").drop_duplicates(subset="episode", keep="last").reset_index(drop=True)

    episodes = df["episode"].values
    rewards = df["avg_reward"].values

    os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(episodes, rewards, linewidth=0.8, alpha=0.3, color="steelblue", label="avg reward")

    # Smoothed trend line (rolling window of 50 data points)
    if len(rewards) >= 50:
        window = 50
        smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
        ax.plot(
            episodes[window - 1:], smoothed,
            linewidth=2, color="red", label=f"smoothed ({window}-pt)",
        )

    # Horizontal break-even line at y=0
    ax.axhline(y=0, color="grey", linestyle="--", linewidth=1, alpha=0.4)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Average Reward")
    last_ep = int(episodes[-1])
    ax.set_title(f"Whist Agent \u2014 Training Reward ({last_ep:,} episodes)")
    ax.legend(loc="lower right", framealpha=0.8)
    ax.grid(True, alpha=0.3, linestyle="--")
    fig.tight_layout()

    path = os.path.join(out_dir, f"reward_ep_{last_ep}.png")
    fig.savefig(path, dpi=100)
    plt.close(fig)
    print(f"Graph saved to {path}")


def plot_winrate(csv_path="winrate.csv", out_dir=GRAPH_DIR):
    """Generate the original win-rate-vs-baseline graph for any run."""
    if not os.path.exists(csv_path):
        print(f"Skipping win-rate graph: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    if df.empty or "episode" not in df.columns or "win_rate_vs_baseline" not in df.columns:
        print(f"Skipping win-rate graph: no usable data in {csv_path}.")
        return

    df = df.sort_values("episode").drop_duplicates("episode", keep="last")
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(
        df["episode"], df["win_rate_vs_baseline"], marker="o", linewidth=1.5,
        color="green", label="win rate vs frozen baseline",
    )
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="50% parity")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Win rate")
    ax.set_ylim(0, 1)
    last_ep = int(df["episode"].iloc[-1])
    ax.set_title(f"Whist Agent - Win Rate vs Frozen Baseline ({last_ep:,} episodes)")
    ax.legend(loc="lower right", framealpha=0.8)
    ax.grid(True, alpha=0.3, linestyle="--")
    fig.tight_layout()
    path = os.path.join(out_dir, f"winrate_ep_{last_ep}.png")
    fig.savefig(path, dpi=100)
    plt.close(fig)
    print(f"Win-rate graph saved to {path}")


def plot_legacy_pair(rewards_csv, winrate_csv, out_dir):
    """Restore the original reward and win-rate graph pair for a run."""
    plot_rewards(rewards_csv, out_dir)
    plot_winrate(winrate_csv, out_dir)


if __name__ == "__main__":
    plot_rewards()
    plot_winrate()
    plot_legacy_pair("rewards_multiseat.csv", "winrate_multiseat.csv", "graphs_multiseat")
    plot_dashboard()
    plot_breakdowns()
