"""Anchor-league training after competitive convergence.

The fixed competitive anchor prevents self-play from drifting away from the
strongest measured policy. Opponents are mixed: anchor, league policies, rule
player, and random legal play.
"""

import atexit
import os
import random
import shutil

os.environ.setdefault("WHIST_RANDOMIZE_LEARNING_SEAT", "1")
os.environ.setdefault("WHIST_SHAPING_SCALE", "0.25")
os.environ.setdefault("WHIST_TEAM_TERMINAL_REWARD", "4.0")

import numpy as np
from sb3_contrib import MaskablePPO

import train
from evaluate import rule_action
from strong_eval import append_checkpoint_evaluation, evaluate_against_rule

LOCK_PATH = "train_anchor_selfplay.lock"
ANCHOR_PATH = os.path.join("checkpoints_competitive", "best_rule.pth")
CHECKPOINT_DIR = "checkpoints_anchor_selfplay"
SCORES_CSV = "anchor_selfplay_checkpoint_scores.csv"
BEST_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "best_rule.pth")
BEST_SCORE = float("-inf")
ANCHOR_PROB = 0.45
LEAGUE_PROB = 0.25
RULE_PROB = 0.20
RANDOM_PROB = 0.10
BASE_POLICY_FN = None
BASE_LEAGUE_FN = None


def acquire_lock():
    try:
        with open(LOCK_PATH, "x", encoding="ascii") as lock:
            lock.write(str(os.getpid()))
    except FileExistsError as exc:
        raise RuntimeError(f"Another anchor run appears active ({LOCK_PATH}).") from exc
    atexit.register(lambda: os.remove(LOCK_PATH) if os.path.exists(LOCK_PATH) else None)


def mixed_policy(model, league_policy, anchor_model):
    def _policy(obs, mask, env):
        roll = random.random()
        if roll < ANCHOR_PROB:
            return train.sample_action(anchor_model, obs, mask)
        if roll < ANCHOR_PROB + LEAGUE_PROB:
            return league_policy(obs, mask)
        if roll < ANCHOR_PROB + LEAGUE_PROB + RULE_PROB:
            return rule_action(env)
        valid = np.flatnonzero(mask)
        return int(env.np_random.choice(valid))
    _policy.uses_env = True
    return _policy


def make_policy(model, anchor_model):
    return mixed_policy(model, BASE_POLICY_FN(model), anchor_model)


def make_league(model, pool_paths, latest_prob, anchor_model):
    return mixed_policy(model, BASE_LEAGUE_FN(model, pool_paths, latest_prob), anchor_model)


def evaluate_anchor(model, anchor_model, episodes, seed):
    """Evaluate the candidate against both teams controlled by the anchor."""
    from whist_env import TEAMS, WhistEnv
    from play import agent_action

    wins = ties = 0
    differences = []
    for index in range(episodes):
        env = WhistEnv()
        env.reset(seed=seed + index)
        candidate_team = index % 2
        while not env.done:
            if TEAMS[env.current_player] == candidate_team:
                action = agent_action(model, env, mcts_sims=0)
            else:
                action = agent_action(anchor_model, env, mcts_sims=0)
            env.step(action)
        difference = env.team_tricks[candidate_team] - env.team_tricks[1 - candidate_team]
        differences.append(difference)
        wins += difference > 0
        ties += difference == 0
    return {
        "games": episodes,
        "wins": int(wins),
        "ties": int(ties),
        "win_rate": wins / episodes,
        "avg_trick_difference": float(np.mean(differences)),
    }


def checkpoint_evaluator(model, checkpoint_path, episode):
    global BEST_SCORE
    rule_result = evaluate_against_rule(model, episodes=200, mcts_sims=0, seed=episode * 23)
    anchor_result = evaluate_anchor(model, ANCHOR_MODEL, episodes=200, seed=episode * 23 + 1)
    append_checkpoint_evaluation(SCORES_CSV, episode, rule_result, anchor_result)
    score = (
        0.5 * rule_result["win_rate"]
        + 0.5 * anchor_result["win_rate"]
        + 0.01 * (rule_result["avg_trick_difference"] + anchor_result["avg_trick_difference"])
    )
    print(
        f"  Rule {rule_result['win_rate']:.1%} ({rule_result['avg_trick_difference']:+.2f}) | "
        f"Anchor {anchor_result['win_rate']:.1%} ({anchor_result['avg_trick_difference']:+.2f})"
    )
    if score > BEST_SCORE:
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        shutil.copy2(checkpoint_path, BEST_CHECKPOINT)
        BEST_SCORE = score
        print(f"  New best anchor-league checkpoint: episode {episode}")


ANCHOR_MODEL = None


def main():
    global BASE_POLICY_FN, BASE_LEAGUE_FN, ANCHOR_MODEL, BEST_SCORE
    acquire_lock()
    if not os.path.exists(ANCHOR_PATH):
        raise FileNotFoundError(f"Missing fixed anchor: {ANCHOR_PATH}")

    ANCHOR_MODEL = MaskablePPO.load(ANCHOR_PATH, device=train.DEVICE)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    seed_checkpoint = os.path.join(CHECKPOINT_DIR, "whist_cp_0.pth")
    if not os.path.exists(seed_checkpoint):
        shutil.copy2(ANCHOR_PATH, seed_checkpoint)

    train.CHECKPOINT_DIR = CHECKPOINT_DIR
    train.REWARDS_CSV = "rewards_anchor_selfplay.csv"
    train.WINRATE_CSV = "winrate_anchor_selfplay.csv"
    train.GRAPH_DIR = "graphs_anchor_selfplay"
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
    train.make_policy_fn = lambda model: make_policy(model, ANCHOR_MODEL)
    train.make_league_policy_fn = lambda model, pool, latest: make_league(
        model, pool, latest, ANCHOR_MODEL
    )
    train.train()


if __name__ == "__main__":
    main()
