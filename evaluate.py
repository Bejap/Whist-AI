"""Repeatable benchmark evaluation for Whist checkpoints.

Examples:
    python evaluate.py --episodes 1000 --opponent all
    python evaluate.py --checkpoint checkpoints/whist_cp_200000.pth \
        --episodes 2000 --opponent rule --mcts-sims 64
"""

import argparse
import csv
import os
import random
from collections import defaultdict

import numpy as np
from sb3_contrib import MaskablePPO

from play import agent_action, random_action
from train import latest_checkpoint
from whist_env import NUM_PLAYERS, TEAMS, WhistEnv

BENCHMARK_CSV = "benchmarks.csv"
BENCHMARK_DETAILS_CSV = "benchmark_games.csv"


def rule_action(env: WhistEnv) -> int:
    """Play a simple fixed strategy that does not use the trained model."""
    valid = np.flatnonzero(env.action_mask()).tolist()
    if len(valid) == 1:
        return int(valid[0])

    if not env.trick_cards:
        non_trumps = [c for c in valid if env.trump_suit == 4 or c // 13 != env.trump_suit]
        return int(min(non_trumps or valid, key=lambda card: card % 13))

    lead_suit = env.trick_cards[0][1] // 13
    current_winner = env._current_trick_winner()
    current_team = TEAMS[current_winner]
    acting_team = TEAMS[env.current_player]
    follow = [c for c in valid if c // 13 == lead_suit]

    if follow:
        candidates = follow
    else:
        candidates = valid

    if current_team == acting_team:
        return int(min(candidates, key=lambda card: card % 13))

    def wins_trick(card):
        projected = env.trick_cards + [(env.current_player, card)]
        return TEAMS[env._peek_trick_winner(projected)] == acting_team

    winning = [card for card in candidates if wins_trick(card)]
    return int(min(winning or candidates, key=lambda card: card % 13))


def load_checkpoint(path=None):
    if path is None:
        path, episode = latest_checkpoint()
    else:
        episode = 0
    if path is None:
        raise FileNotFoundError("No checkpoint found. Train the agent first.")
    return MaskablePPO.load(path, device=os.getenv("WHIST_DEVICE", "auto")), path, episode


def play_benchmark_game(model, opponent, seed, agent_team, mcts_sims):
    env = WhistEnv()
    env.reset(seed=seed)
    actions = 0
    while not env.done:
        player = env.current_player
        if TEAMS[player] == agent_team:
            action = agent_action(model, env, mcts_sims=mcts_sims)
        elif opponent == "random":
            action = random_action(env)
        else:
            action = rule_action(env)
        env.step(action)
        actions += 1

    agent_tricks = env.team_tricks[agent_team]
    opponent_tricks = env.team_tricks[1 - agent_team]
    return {
        "agent_team": agent_team,
        "agent_tricks": agent_tricks,
        "opponent_tricks": opponent_tricks,
        "trick_difference": agent_tricks - opponent_tricks,
        "win": int(agent_tricks > opponent_tricks),
        "tie": int(agent_tricks == opponent_tricks),
        "actions": actions,
        "trump_suit": int(env.trump_suit),
    }


def evaluate(model, opponent, episodes, mcts_sims, seed):
    rows = []
    for index in range(episodes):
        rows.append(
            play_benchmark_game(
                model,
                opponent,
                seed + index,
                agent_team=index % 2,
                mcts_sims=mcts_sims,
            )
        )
    return rows


def write_results(path, checkpoint, checkpoint_episode, opponent, rows, mcts_sims):
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as file:
        fields = [
            "checkpoint", "checkpoint_episode", "opponent", "mcts_sims",
            "games", "wins", "ties", "win_rate", "tie_rate",
            "avg_agent_tricks", "avg_opponent_tricks", "avg_trick_difference",
        ]
        writer = csv.DictWriter(file, fieldnames=fields)
        if write_header:
            writer.writeheader()
        games = len(rows)
        writer.writerow({
            "checkpoint": checkpoint,
            "checkpoint_episode": checkpoint_episode,
            "opponent": opponent,
            "mcts_sims": mcts_sims,
            "games": games,
            "wins": sum(row["win"] for row in rows),
            "ties": sum(row["tie"] for row in rows),
            "win_rate": f"{sum(row['win'] for row in rows) / games:.6f}",
            "tie_rate": f"{sum(row['tie'] for row in rows) / games:.6f}",
            "avg_agent_tricks": f"{np.mean([row['agent_tricks'] for row in rows]):.4f}",
            "avg_opponent_tricks": f"{np.mean([row['opponent_tricks'] for row in rows]):.4f}",
            "avg_trick_difference": f"{np.mean([row['trick_difference'] for row in rows]):.4f}",
        })


def write_game_details(path, checkpoint, checkpoint_episode, opponent, rows, mcts_sims):
    """Persist one row per game for seat and trump-suit analysis."""
    write_header = not os.path.exists(path)
    fields = [
        "checkpoint", "checkpoint_episode", "opponent", "mcts_sims",
        "agent_team", "trump_suit", "agent_tricks", "opponent_tricks",
        "trick_difference", "win", "tie",
    ]
    with open(path, "a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({
                "checkpoint": checkpoint,
                "checkpoint_episode": checkpoint_episode,
                "opponent": opponent,
                "mcts_sims": mcts_sims,
                **{field: row[field] for field in fields[4:]},
            })


def main():
    parser = argparse.ArgumentParser(description="Benchmark a Whist checkpoint.")
    parser.add_argument("--checkpoint", help="Checkpoint path; defaults to latest.")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--opponent", choices=["random", "rule", "all"], default="all")
    parser.add_argument("--mcts-sims", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--output", default=BENCHMARK_CSV)
    parser.add_argument("--details-output", default=BENCHMARK_DETAILS_CSV)
    args = parser.parse_args()

    model, checkpoint, checkpoint_episode = load_checkpoint(args.checkpoint)
    opponents = ["random", "rule"] if args.opponent == "all" else [args.opponent]
    for opponent in opponents:
        rows = evaluate(model, opponent, args.episodes, max(0, args.mcts_sims), args.seed)
        write_results(args.output, checkpoint, checkpoint_episode, opponent, rows, max(0, args.mcts_sims))
        write_game_details(
            args.details_output, checkpoint, checkpoint_episode, opponent,
            rows, max(0, args.mcts_sims),
        )
        wins = sum(row["win"] for row in rows)
        ties = sum(row["tie"] for row in rows)
        avg_diff = np.mean([row["trick_difference"] for row in rows])
        print(
            f"{opponent}: {wins / len(rows):.1%} wins, "
            f"{ties / len(rows):.1%} ties, avg trick difference {avg_diff:.2f}"
        )


if __name__ == "__main__":
    main()
