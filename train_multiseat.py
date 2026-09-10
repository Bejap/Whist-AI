"""Launch the multi-seat shared-policy training experiment.

This keeps the original run intact and writes the experiment to separate
checkpoint and metric paths. Configure the duration with WHIST_TOTAL_EPISODES.
"""

import os
import atexit

import train


LOCK_PATH = "train_multiseat.lock"


def acquire_lock():
    try:
        lock = open(LOCK_PATH, "x", encoding="ascii")
    except FileExistsError as exc:
        raise RuntimeError(
            f"Another multi-seat run appears to be active ({LOCK_PATH}). "
            "Stop it before starting a second run."
        ) from exc
    lock.write(str(os.getpid()))
    lock.close()
    atexit.register(lambda: os.remove(LOCK_PATH) if os.path.exists(LOCK_PATH) else None)


def main():
    acquire_lock()
    train.CHECKPOINT_DIR = os.getenv(
        "WHIST_EXPERIMENT_CHECKPOINT_DIR", os.path.join("checkpoints", "multiseat")
    )
    train.REWARDS_CSV = os.getenv(
        "WHIST_EXPERIMENT_REWARDS_CSV", os.path.join("graphs", "benchmarks", "rewards_multiseat.csv")
    )
    train.WINRATE_CSV = os.getenv(
        "WHIST_EXPERIMENT_WINRATE_CSV", os.path.join("graphs", "benchmarks", "winrate_multiseat.csv")
    )
    train.GRAPH_DIR = os.getenv(
        "WHIST_EXPERIMENT_GRAPH_DIR", os.path.join("graphs", "multiseat")
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