"""Fast competitive training: MLP multi-seat PPO plus rule-player opponents."""

import atexit
import os
import random
import shutil

os.environ.setdefault("WHIST_RANDOMIZE_LEARNING_SEAT", "1")
os.environ.setdefault("WHIST_SHAPING_SCALE", "0.25")
os.environ.setdefault("WHIST_TEAM_TERMINAL_REWARD", "4.0")

import train
from evaluate import rule_action
from strong_eval import append_checkpoint_evaluation, evaluate_against_rule

LOCK_PATH = "train_competitive.lock"
SCORES_CSV = os.path.join("graphs", "benchmarks", "competitive_checkpoint_scores.csv")
BEST_CHECKPOINT = os.path.join("checkpoints", "competitive", "best_rule.pth")
BEST_SCORE = float("-inf")
RULE_OPPONENT_PROB = 0.35
BASE_POLICY_FN = None
BASE_LEAGUE_FN = None


def acquire_lock():
    try:
        with open(LOCK_PATH, "x", encoding="ascii") as lock:
            lock.write(str(os.getpid()))
    except FileExistsError as exc:
        raise RuntimeError(f"Another competitive run appears active ({LOCK_PATH}).") from exc
    atexit.register(lambda: os.remove(LOCK_PATH) if os.path.exists(LOCK_PATH) else None)


def mix_policy(league_policy, model, obs, mask):
    if random.random() < RULE_OPPONENT_PROB:
        return rule_action_from_env
    return league_policy(obs, mask)


def rule_action_from_env(obs, mask, env):
    return rule_action(env)


def mixed_policy(league_policy):
    def _policy(obs, mask, env):
        if random.random() < RULE_OPPONENT_PROB:
            return rule_action(env)
        return league_policy(obs, mask)
    _policy.uses_env = True
    return _policy


def make_competitive_league(model, pool_paths, latest_prob):
    league = BASE_LEAGUE_FN(model, pool_paths, latest_prob)
    return mixed_policy(league)


def make_competitive_policy(model):
    league = BASE_POLICY_FN(model)
    return mixed_policy(league)


def checkpoint_evaluator(model, checkpoint_path, episode):
    global BEST_SCORE
    raw = evaluate_against_rule(model, episodes=200, mcts_sims=0, seed=episode * 17)
    append_checkpoint_evaluation(SCORES_CSV, episode, raw, raw)
    print(
        f"  Rule benchmark | {raw['win_rate']:.1%} wins | "
        f"{raw['avg_trick_difference']:+.2f} tricks"
    )
    score = raw["win_rate"] + 0.02 * raw["avg_trick_difference"]
    if score > BEST_SCORE:
        os.makedirs(os.path.dirname(BEST_CHECKPOINT), exist_ok=True)
        shutil.copy2(checkpoint_path, BEST_CHECKPOINT)
        BEST_SCORE = score
        print(f"  New best competitive checkpoint: episode {episode}")


def main():
    global BASE_POLICY_FN, BASE_LEAGUE_FN, BEST_SCORE
    acquire_lock()
    train.CHECKPOINT_DIR = os.path.join("checkpoints", "competitive")
    train.REWARDS_CSV = os.path.join("graphs", "benchmarks", "rewards_competitive.csv")
    train.WINRATE_CSV = os.path.join("graphs", "benchmarks", "winrate_competitive.csv")
    train.GRAPH_DIR = os.path.join("graphs", "competitive")
    train.BASELINE_CHECKPOINT = os.path.join(train.CHECKPOINT_DIR, "baseline.pth")
    train.TOTAL_EPISODES = int(os.getenv("WHIST_TOTAL_EPISODES", "1000000"))
    train.CHECKPOINT_EVERY = int(os.getenv("WHIST_CHECKPOINT_EVERY", "10000"))
    train.GRAPH_EVERY = int(os.getenv("WHIST_GRAPH_EVERY", "10000"))
    train.LOG_EVERY = int(os.getenv("WHIST_LOG_EVERY", "250"))
    train.EVAL_EVERY = 0
    train.POLICY_KWARGS = {"net_arch": [192, 192, 128]}
    train.CHECKPOINT_EVALUATOR = checkpoint_evaluator
    BASE_POLICY_FN = train.make_policy_fn
    BASE_LEAGUE_FN = train.make_league_policy_fn
    train.make_policy_fn = make_competitive_policy
    train.make_league_policy_fn = make_competitive_league
    train.train()


if __name__ == "__main__":
    main()
