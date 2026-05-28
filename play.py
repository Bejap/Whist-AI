"""Watch a trained Whist agent play a round."""

import os
import sys

import numpy as np
import torch

from whist_env import WhistEnv, card_name, trump_name, SUITS, TEAMS

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_model():
    """Load the latest checkpoint."""
    from train import latest_checkpoint
    from stable_baselines3 import PPO

    ckpt_path, episode = latest_checkpoint()
    if ckpt_path is None:
        print("No checkpoint found. Train the agent first with: python train.py")
        sys.exit(1)

    print(f"Loading checkpoint: {ckpt_path} (episode {episode})")
    model = PPO.load(ckpt_path, device="cpu")
    return model


def agent_action(model, obs, mask):
    """Pick an action using the trained model with action masking."""
    obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        logits = model.policy.get_distribution(obs_t).distribution.logits
    logits = logits.squeeze(0).numpy()
    logits[mask == 0] = -1e8
    # Greedy (argmax) for demonstration
    return int(np.argmax(logits))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def play():
    model = load_model()
    env = WhistEnv(render_mode="human")
    obs, info = env.reset()

    print("=" * 50)
    print(f"  WHIST — Trump: {trump_name(env.trump_suit)}")
    print(f"  Team 0 (Players 0, 2) vs Team 1 (Players 1, 3)")
    print("=" * 50)

    while True:
        env.render()

        mask = env.action_mask()
        obs_current = env._get_obs()
        action = agent_action(model, obs_current, mask)

        player = env.current_player
        print(f"  → Player {player} plays: {card_name(action)}")

        obs, reward, terminated, truncated, info = env.step(action)

        if len(env.trick_cards) == 0 and not terminated:
            # Trick just resolved
            print(f"  ★ Trick won! Score: {env.team_tricks}")
            input("  Press Enter for next trick...")

        if terminated:
            break

    print("\n" + "=" * 50)
    print(f"  FINAL SCORE")
    print(f"  Team 0 (Players 0, 2): {env.team_tricks[0]} tricks")
    print(f"  Team 1 (Players 1, 3): {env.team_tricks[1]} tricks")
    winner = 0 if env.team_tricks[0] > env.team_tricks[1] else 1
    print(f"  🏆 Team {winner} wins!")
    print("=" * 50)


if __name__ == "__main__":
    play()
