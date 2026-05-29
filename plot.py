"""Standalone plotting script – reads rewards.csv and produces a reward graph."""

import csv
import os
import sys

import numpy as np
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

    episodes, rewards = [], []
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        for row in reader:
            if len(row) >= 2:
                episodes.append(int(row[0]))
                rewards.append(float(row[1]))

    if not episodes:
        print("No data found in rewards.csv.")
        sys.exit(1)

    os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(episodes, rewards, linewidth=0.8, alpha=0.6, label="avg reward")

    # Smoothed trend line (rolling window of 50 data points)
    if len(rewards) >= 50:
        window = 50
        smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
        ax.plot(
            episodes[window - 1:], smoothed,
            linewidth=2, color="red", label=f"smoothed ({window}-pt)",
        )

    ax.set_xlabel("Episode")
    ax.set_ylabel("Average Reward")
    ax.set_title(f"Training Reward (up to episode {episodes[-1]})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = os.path.join(out_dir, f"reward_ep_{episodes[-1]}.png")
    fig.savefig(path, dpi=100)
    plt.close(fig)
    print(f"Graph saved to {path}")


if __name__ == "__main__":
    plot_rewards()
