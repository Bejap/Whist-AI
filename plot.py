"""Standalone plotting script – reads rewards.csv and produces a reward graph."""

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


if __name__ == "__main__":
    plot_rewards()
