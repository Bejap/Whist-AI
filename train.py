"""Training script for Whist RL agent using PPO with self-play."""

import csv
import glob
import os
import re
import sys

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm

from whist_env import SelfPlayWrapper, WhistEnv, TEAMS

# ---------------------------------------------------------------------------
# Configuration – optimised for low-end hardware (~4 GB RAM, CPU only)
# ---------------------------------------------------------------------------
CHECKPOINT_DIR = "checkpoints"
REWARDS_CSV = "rewards.csv"
TOTAL_EPISODES = 100_000
CHECKPOINT_EVERY = 1000
LOG_EVERY = 500
KEEP_CHECKPOINTS = 5

# PPO hyper-parameters (small footprint)
PPO_KWARGS = dict(
    learning_rate=3e-4,
    n_steps=512,          # rollout buffer length per update
    batch_size=64,
    n_epochs=4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    verbose=0,
    device="cpu",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def latest_checkpoint():
    """Return (path, episode) of the most recent checkpoint, or (None, 0)."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    # SB3 may save as .pth or .pth.zip depending on version
    files = glob.glob(os.path.join(CHECKPOINT_DIR, "whist_cp_*.pth"))
    files += glob.glob(os.path.join(CHECKPOINT_DIR, "whist_cp_*.pth.zip"))
    if not files:
        return None, 0
    # Extract episode number from filename
    def _ep(f):
        m = re.search(r"whist_cp_(\d+)\.pth", f)
        return int(m.group(1)) if m else 0
    files.sort(key=_ep)
    best = files[-1]
    return best, _ep(best)


def save_checkpoint(model, episode):
    """Save model and prune old checkpoints."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    path = os.path.join(CHECKPOINT_DIR, f"whist_cp_{episode}.pth")
    model.save(path)  # SB3 appends .zip automatically

    # Prune – keep only the newest KEEP_CHECKPOINTS files
    files = glob.glob(os.path.join(CHECKPOINT_DIR, "whist_cp_*.pth"))
    files += glob.glob(os.path.join(CHECKPOINT_DIR, "whist_cp_*.pth.zip"))
    # Deduplicate
    files = sorted(set(files), key=os.path.getmtime)
    while len(files) > KEEP_CHECKPOINTS:
        os.remove(files.pop(0))


def append_reward(episode, reward):
    """Append a row to rewards.csv."""
    write_header = not os.path.exists(REWARDS_CSV)
    with open(REWARDS_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["episode", "avg_reward"])
        writer.writerow([episode, f"{reward:.4f}"])


def make_policy_fn(model):
    """Create a policy function for self-play opponents."""
    def _policy(obs, mask):
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits = model.policy.get_distribution(obs_t).distribution.logits
        # Mask invalid actions
        logits = logits.squeeze(0).numpy()
        logits[mask == 0] = -1e8
        probs = np.exp(logits - logits.max())
        probs = probs / probs.sum()
        return int(np.random.choice(len(probs), p=probs))
    return _policy


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train():
    # Resume from checkpoint if available
    ckpt_path, start_episode = latest_checkpoint()

    env = SelfPlayWrapper(WhistEnv())

    if ckpt_path is not None:
        print(f"► Resuming from checkpoint: {ckpt_path} (episode {start_episode})")
        model = PPO.load(ckpt_path, env=env, **PPO_KWARGS)
    else:
        print("► Starting fresh training (episode 0)")
        start_episode = 0
        model = PPO("MlpPolicy", env, **PPO_KWARGS)

    # Wire self-play policy
    env.set_policy(make_policy_fn(model))

    remaining = TOTAL_EPISODES - start_episode
    if remaining <= 0:
        print("Training already complete.")
        return

    episode = start_episode
    reward_buffer = []

    pbar = tqdm(total=remaining, desc="Training", unit="ep", file=sys.stdout)

    while episode < TOTAL_EPISODES:
        obs, info = env.reset()
        episode_reward = 0.0
        done = False

        while not done:
            mask = env.env.action_mask()
            # Get action from model with masking
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                logits = model.policy.get_distribution(obs_t).distribution.logits
            logits = logits.squeeze(0).numpy()
            logits[mask == 0] = -1e8
            probs = np.exp(logits - logits.max())
            probs = probs / probs.sum()
            action = int(np.random.choice(len(probs), p=probs))

            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated

        reward_buffer.append(episode_reward)
        episode += 1
        pbar.update(1)

        # Periodically train the model on collected experience
        if episode % 32 == 0:
            # Run a small number of PPO learn steps
            env.set_policy(make_policy_fn(model))
            model.learn(total_timesteps=512, reset_num_timesteps=False)
            env.set_policy(make_policy_fn(model))

        # Logging
        if episode % LOG_EVERY == 0 and reward_buffer:
            avg_r = np.mean(reward_buffer[-LOG_EVERY:])
            append_reward(episode, avg_r)
            pbar.set_postfix(avg_reward=f"{avg_r:.2f}")

        # Checkpoint
        if episode % CHECKPOINT_EVERY == 0:
            save_checkpoint(model, episode)
            tqdm.write(f"  💾 Checkpoint saved at episode {episode}")

    pbar.close()
    # Final save
    save_checkpoint(model, episode)
    print(f"\nTraining complete. Final episode: {episode}")


if __name__ == "__main__":
    train()
