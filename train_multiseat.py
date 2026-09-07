"""Launch the multi-seat shared-policy training experiment.

This keeps the original run intact and writes the experiment to separate
checkpoint and metric paths. Configure the duration with WHIST_TOTAL_EPISODES.
"""

import os

import train


def main():
    train.CHECKPOINT_DIR = os.getenv(
        "WHIST_EXPERIMENT_CHECKPOINT_DIR", "checkpoints_multiseat"
    )
    train.REWARDS_CSV = os.getenv(
        "WHIST_EXPERIMENT_REWARDS_CSV", "rewards_multiseat.csv"
    )
    train.WINRATE_CSV = os.getenv(
        "WHIST_EXPERIMENT_WINRATE_CSV", "winrate_multiseat.csv"
    )
    train.GRAPH_DIR = os.getenv(
        "WHIST_EXPERIMENT_GRAPH_DIR", "graphs_multiseat"
    )
    train.BASELINE_CHECKPOINT = os.path.join(
        train.CHECKPOINT_DIR, "baseline.pth"
    )
    train.TOTAL_EPISODES = int(
        os.getenv("WHIST_TOTAL_EPISODES", str(train.TOTAL_EPISODES))
    )
    train.CHECKPOINT_EVERY = int(
        os.getenv("WHIST_CHECKPOINT_EVERY", str(train.CHECKPOINT_EVERY))
    )
    train.GRAPH_EVERY = int(
        os.getenv("WHIST_GRAPH_EVERY", str(train.GRAPH_EVERY))
    )
    train.EVAL_EVERY = int(
        os.getenv("WHIST_EVAL_EVERY", str(train.EVAL_EVERY))
    )
    train.EVAL_EPISODES = int(
        os.getenv("WHIST_EVAL_EPISODES", str(train.EVAL_EPISODES))
    )
    train.KEEP_CHECKPOINTS = max(train.KEEP_CHECKPOINTS, train.LEAGUE_POOL_SIZE)
    os.environ["WHIST_RANDOMIZE_LEARNING_SEAT"] = "1"
    train.train()


if __name__ == "__main__":
    main()