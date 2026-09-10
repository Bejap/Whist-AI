"""Post-competitive league self-play training.

Starts from the best competitive checkpoint and trains in isolated output
paths. The opponent mixture is current/historical policy, rule player, and
random legal play.
"""

import atexit
import os
import random
import shutil

os.environ.setdefault("WHIST_RANDOMIZE_LEARNING_SEAT", "1")
os.environ.setdefault("WHIST_SHAPING_SCALE", "0.25")
os.environ.setdefault("WHIST_TEAM_TERMINAL_REWARD", "4.0")

import numpy as np

import train
from evaluate import rule_action
from strong_eval import append_checkpoint_evaluation, evaluate_against_rule

LOCK_PATH = "train_post_selfplay.lock"
SOURCE_CHECKPOINT = os.path.join("checkpoints", "competitive", "best_rule.pth")
CHECKPOINT_DIR = os.path.join("checkpoints", "post_selfplay")
SCORES_CSV = os.path.join("graphs", "benchmarks", "post_selfplay_checkpoint_scores.csv")
BEST_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "best_rule.pth")
BEST_SCORE = float("-inf")
RULE_PROB = 0.20
RANDOM_PROB = 0.10
BASE_POLICY_FN = None
BASE_LEAGUE_FN = None


def acquire_lock():
    try:
        with open(LOCK_PATH, "x", encoding="ascii") as lock:
            lock.write(str(os.getpid()))
    except FileExistsError as exc:
        raise RuntimeError(f"Another post-self-play run appears active ({LOCK_PATH}).") from exc
    atexit.register(lambda: os.remove(LOCK_PATH) if os.path.exists(LOCK_PATH) else None)


def mixed_policy(league_policy):
    def _policy(obs, mask, env):
        roll = random.random()
        if roll < RULE_PROB:
            return rule_action(env)
        if roll < RULE_PROB + RANDOM_PROB:
            valid = np.flatnonzero(mask)
            return int(env.np_random.choice(valid))
        return league_policy(obs, mask)
    _policy.uses_env = True
    return _policy


def make_policy(model):
    return mixed_policy(BASE_POLICY_FN(model))


def make_league(model, pool_paths, latest_prob):
    return mixed_policy(BASE_LEAGUE_FN(model, pool_paths, latest_prob))


def checkpoint_evaluator(model, checkpoint_path, episode):
    global BEST_SCORE
    result = evaluate_against_rule(model, episodes=200, mcts_sims=0, seed=episode * 19)
    append_checkpoint_evaluation(SCORES_CSV, episode, result, result)
    score = result["win_rate"] + 0.02 * result["avg_trick_difference"]
    print(
        f"  Rule benchmark | {result['win_rate']:.1%} wins | "
        f"{result['avg_trick_difference']:+.2f} tricks"
    )
    if score > BEST_SCORE:
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        shutil.copy2(checkpoint_path, BEST_CHECKPOINT)
        BEST_SCORE = score
        print(f"  New best post-self-play checkpoint: episode {episode}")


def main():
    global BASE_POLICY_FN, BASE_LEAGUE_FN, BEST_SCORE
    acquire_lock()
    if not os.path.exists(SOURCE_CHECKPOINT):
        raise FileNotFoundError(
            f"Missing source model: {SOURCE_CHECKPOINT}. Finish competitive training first."
        )

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    seed_checkpoint = os.path.join(CHECKPOINT_DIR, "whist_cp_0.pth")
    if not os.path.exists(seed_checkpoint):
        shutil.copy2(SOURCE_CHECKPOINT, seed_checkpoint)

    train.CHECKPOINT_DIR = CHECKPOINT_DIR
    train.REWARDS_CSV = os.path.join("graphs", "benchmarks", "rewards_post_selfplay.csv")
    train.WINRATE_CSV = os.path.join("graphs", "benchmarks", "winrate_post_selfplay.csv")
    train.GRAPH_DIR = os.path.join("graphs", "post_selfplay")
    train.BASELINE_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "baseline.pth")
    train.TOTAL_EPISODES = int(os.getenv("WHIST_TOTAL_EPISODES", "1000000"))
    train.CHECKPOINT_EVERY = int(os.getenv("WHIST_CHECKPOINT_EVERY", "10000"))
    train.GRAPH_EVERY = int(os.getenv("WHIST_GRAPH_EVERY", "10000"))
    train.LOG_EVERY = int(os.getenv("WHIST_LOG_EVERY", "250"))
    train.EVAL_EVERY = 0
    train.POLICY_KWARGS = {"net_arch": [192, 192, 128]}
    train.CHECKPOINT_EVALUATOR = checkpoint_evaluator
    BASE_POLICY_FN = train.make_policy_fn
    BASE_LEAGUE_FN = train.make_league_policy_fn
    train.make_policy_fn = make_policy
    train.make_league_policy_fn = make_league
    train.train()


if __name__ == "__main__":
    main()
