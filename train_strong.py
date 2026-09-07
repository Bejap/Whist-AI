"""Strong Whist experiment with a transformer and rule-based promotion."""

import atexit
import csv
import os
import shutil

os.environ.setdefault("WHIST_RANDOMIZE_LEARNING_SEAT", "1")
os.environ.setdefault("WHIST_SHAPING_SCALE", "0.15")
os.environ.setdefault("WHIST_TEAM_TERMINAL_REWARD", "8.0")

import train
from models import TransformerCardExtractor
from strong_eval import append_checkpoint_evaluation, evaluate_against_rule

LOCK_PATH = "train_strong.lock"
SCORES_CSV = "strong_checkpoint_scores.csv"
BEST_CHECKPOINT = "checkpoints_strong/best_rule.pth"
BEST_SCORE = float("-inf")


def acquire_lock():
    try:
        with open(LOCK_PATH, "x", encoding="ascii") as lock:
            lock.write(str(os.getpid()))
    except FileExistsError as exc:
        raise RuntimeError(f"Another strong run appears active ({LOCK_PATH}).") from exc
    atexit.register(lambda: os.remove(LOCK_PATH) if os.path.exists(LOCK_PATH) else None)


def checkpoint_evaluator(model, checkpoint_path, episode):
    global BEST_SCORE
    raw = evaluate_against_rule(model, episodes=50, mcts_sims=0, seed=episode * 10)
    search = evaluate_against_rule(model, episodes=4, mcts_sims=4, seed=episode * 10 + 1)
    append_checkpoint_evaluation(SCORES_CSV, episode, raw, search)
    print(
        f"  Rule benchmark | raw {raw['win_rate']:.1%} ({raw['avg_trick_difference']:+.2f} tricks)"
        f" | hidden-hand MCTS {search['win_rate']:.1%} ({search['avg_trick_difference']:+.2f})"
    )
    score = raw["win_rate"] + 0.02 * raw["avg_trick_difference"]
    if score > BEST_SCORE:
        os.makedirs(os.path.dirname(BEST_CHECKPOINT), exist_ok=True)
        shutil.copy2(checkpoint_path, BEST_CHECKPOINT)
        BEST_SCORE = score
        print(f"  New best rule checkpoint: episode {episode}")


def load_best_score():
    """Restore the best raw-policy score when a strong run resumes."""
    if not os.path.exists(SCORES_CSV):
        return float("-inf")
    best = float("-inf")
    with open(SCORES_CSV, newline="") as file:
        for row in csv.DictReader(file):
            if row.get("mode") == "raw":
                score = float(row["win_rate"]) + 0.02 * float(row["avg_trick_difference"])
                best = max(best, score)
    return best


def main():
    global BEST_SCORE
    acquire_lock()
    train.CHECKPOINT_DIR = "checkpoints_strong"
    train.REWARDS_CSV = "rewards_strong.csv"
    train.WINRATE_CSV = "winrate_strong.csv"
    train.GRAPH_DIR = "graphs_strong"
    train.BASELINE_CHECKPOINT = os.path.join(train.CHECKPOINT_DIR, "baseline.pth")
    train.TOTAL_EPISODES = int(os.getenv("WHIST_TOTAL_EPISODES", "1000000"))
    train.CHECKPOINT_EVERY = int(os.getenv("WHIST_CHECKPOINT_EVERY", "5000"))
    train.GRAPH_EVERY = int(os.getenv("WHIST_GRAPH_EVERY", "10000"))
    train.LOG_EVERY = int(os.getenv("WHIST_LOG_EVERY", "250"))
    train.EVAL_EVERY = 0
    train.POLICY_KWARGS = {
        "features_extractor_class": TransformerCardExtractor,
        "features_extractor_kwargs": {"features_dim": 256},
        "net_arch": {"pi": [256, 128], "vf": [256, 128]},
    }
    train.CHECKPOINT_EVALUATOR = checkpoint_evaluator
    BEST_SCORE = load_best_score()
    train.train()


if __name__ == "__main__":
    main()