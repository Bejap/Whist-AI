"""Fixed-opponent evaluation used to select strong Whist checkpoints."""

import csv
import os

import numpy as np

from evaluate import rule_action
from play import agent_action
from whist_env import TEAMS, WhistEnv


def play_team_game(model, seed, agent_team, mcts_sims):
    """Play one balanced team match: model controls both seats on agent_team."""
    env = WhistEnv()
    env.reset(seed=seed)
    while not env.done:
        if TEAMS[env.current_player] == agent_team:
            action = agent_action(model, env, mcts_sims=mcts_sims)
        else:
            action = rule_action(env)
        env.step(action)

    agent_tricks = env.team_tricks[agent_team]
    opponent_tricks = env.team_tricks[1 - agent_team]
    return agent_tricks, opponent_tricks


def evaluate_against_rule(model, episodes, mcts_sims, seed):
    """Evaluate all four seats by alternating which complete team uses the model."""
    wins = ties = 0
    differences = []
    for index in range(episodes):
        agent_tricks, opponent_tricks = play_team_game(
            model, seed + index, agent_team=index % 2, mcts_sims=mcts_sims
        )
        wins += agent_tricks > opponent_tricks
        ties += agent_tricks == opponent_tricks
        differences.append(agent_tricks - opponent_tricks)
    return {
        "games": episodes,
        "wins": int(wins),
        "ties": int(ties),
        "win_rate": wins / episodes,
        "avg_trick_difference": float(np.mean(differences)),
    }


def append_checkpoint_evaluation(path, episode, raw, search):
    """Append raw-policy and hidden-hand-search results for one checkpoint."""
    write_header = not os.path.exists(path)
    fields = [
        "episode", "mode", "games", "wins", "ties", "win_rate",
        "avg_trick_difference",
    ]
    with open(path, "a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        if write_header:
            writer.writeheader()
        for mode, result in (("raw", raw), ("hidden_hand_mcts", search)):
            writer.writerow({"episode": episode, "mode": mode, **result})
