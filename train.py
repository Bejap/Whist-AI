"""Training script for Whist RL agent using PPO with self-play."""

import csv
import glob
import os
import random
import re
import shutil
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm

from models import TransformerCardExtractor
from whist_env import SelfPlayWrapper, WhistEnv

# ---------------------------------------------------------------------------
# Configuration – optimised for low-end hardware (~4 GB RAM, CPU only)
# ---------------------------------------------------------------------------
CHECKPOINT_DIR = "checkpoints"
REWARDS_CSV = "rewards.csv"
GRAPH_DIR = "graphs"
TOTAL_EPISODES = 100_000
CHECKPOINT_EVERY = 10_000
GRAPH_EVERY = 25_000
LOG_EVERY = 500
KEEP_CHECKPOINTS = 10

# PPO hyper-parameters (small footprint)
# learning_rate is set as a schedule below; ent_coef is decayed manually
# in the EpisodeTracker callback.
PPO_KWARGS = dict(
    n_steps=512,          # rollout buffer length per update
    batch_size=64,
    n_epochs=4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    verbose=0,
    device="cpu",
)

# Approximate timesteps per episode (13 tricks, 1 action per trick for the
# learning player in the self-play wrapper).
STEPS_PER_EPISODE = 13

# League self-play settings
LEAGUE_POOL_SIZE = 5        # number of checkpoints to keep in the opponent pool
LEAGUE_LATEST_PROB = 0.70   # probability of using the latest policy
OPPONENT_EPSILON_START = 0.20
OPPONENT_EPSILON_END = 0.03

POLICY_KWARGS = dict(
    features_extractor_class=TransformerCardExtractor,
    features_extractor_kwargs=dict(
        card_embed_dim=64,
        nhead=4,
        num_layers=2,
        features_dim=256,
        dropout=0.1,
    ),
)

# Entropy coefficient decay parameters (decayed manually in EpisodeTracker)
ENT_COEF_START = 0.01
ENT_COEF_END = 0.001


# ---------------------------------------------------------------------------
# Schedules
# ---------------------------------------------------------------------------

def linear_schedule(start: float, end: float):
    """Return a callable that linearly decays from start to end.

    The callable receives `progress_remaining` (1.0 → 0.0) from SB3.
    """
    def _schedule(progress_remaining: float) -> float:
        return end + (start - end) * progress_remaining
    return _schedule


LR_SCHEDULE = linear_schedule(3e-4, 5e-5)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def handle_fresh_start():
    """If fresh_start.flag exists, delete all checkpoints and rewards.csv."""
    flag = "fresh_start.flag"
    if not os.path.exists(flag):
        return
    print("► fresh_start.flag detected — wiping checkpoints and rewards.csv")
    # Delete checkpoints directory
    if os.path.isdir(CHECKPOINT_DIR):
        shutil.rmtree(CHECKPOINT_DIR)
    # Delete rewards.csv
    if os.path.exists(REWARDS_CSV):
        os.remove(REWARDS_CSV)
    # Delete the flag itself
    os.remove(flag)
    print("  ✓ Clean slate ready")

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


def get_checkpoint_pool():
    """Return a list of up to LEAGUE_POOL_SIZE most recent checkpoint paths."""
    files = glob.glob(os.path.join(CHECKPOINT_DIR, "whist_cp_*.pth"))
    files += glob.glob(os.path.join(CHECKPOINT_DIR, "whist_cp_*.pth.zip"))
    if not files:
        return []
    def _ep(f):
        m = re.search(r"whist_cp_(\d+)\.pth", f)
        return int(m.group(1)) if m else 0
    files = sorted(set(files), key=_ep)
    return files[-LEAGUE_POOL_SIZE:]


def save_checkpoint(model, episode):
    """Save model and prune old checkpoints."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    path = os.path.join(CHECKPOINT_DIR, f"whist_cp_{episode}.pth")
    model.save(path)

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


def save_reward_graph():
    """Read rewards.csv and save a reward-over-time graph to graphs/."""
    if not os.path.exists(REWARDS_CSV):
        return
    episodes, rewards = [], []
    with open(REWARDS_CSV, newline="") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        for row in reader:
            if len(row) >= 2:
                episodes.append(int(row[0]))
                rewards.append(float(row[1]))
    if not episodes:
        return

    os.makedirs(GRAPH_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(episodes, rewards, linewidth=0.8, alpha=0.6, label="avg reward")

    # Add a smoothed trend line (rolling window of 50 log entries)
    if len(rewards) >= 50:
        window = 50
        smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
        ax.plot(
            episodes[window - 1:], smoothed,
            linewidth=2, color="red", label=f"smoothed ({window}-pt)",
        )

    ax.set_xlabel("Episode")
    ax.set_ylabel("Average Reward")
    ax.set_title(f"Training Reward (up to episode {episodes[-1]})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(GRAPH_DIR, f"reward_ep_{episodes[-1]}.png")
    fig.savefig(path, dpi=100)
    plt.close(fig)


def sample_action(model, obs, mask):
    """Sample an action from the model with action masking."""
    action, _ = model.predict(obs, deterministic=False)
    action = int(action)
    if mask[action] > 0:
        return action
    valid = np.where(mask > 0)[0]
    if len(valid) == 0:
        return 0
    return int(np.random.choice(valid))


def make_policy_fn(model):
    """Create a policy function for self-play opponents."""
    def _policy(obs, mask):
        return sample_action(model, obs, mask)
    return _policy


def make_league_policy_fn(model, pool_paths):
    """Create a league-based policy function.

    - 70% chance: use the latest (live) model
    - 30% chance: use a randomly chosen older checkpoint from the pool

    Older checkpoints are loaded lazily and cached to avoid repeated I/O.
    """
    # Cache for loaded opponent policies (path -> model)
    _cache = {}

    def _load_opponent(path):
        if path not in _cache:
            try:
                # Load policy parameters only (no env needed for inference)
                _cache[path] = RecurrentPPO.load(path, device="cpu")
            except Exception:
                _cache[path] = None
        return _cache[path]

    def _policy(obs, mask):
        # League selection
        if len(pool_paths) <= 1:
            return sample_action(model, obs, mask)
        use_latest = random.random() < LEAGUE_LATEST_PROB
        if use_latest:
            return sample_action(model, obs, mask)
        else:
            # Pick from older checkpoints (all except the last/latest)
            older = pool_paths[:-1]
            chosen_path = random.choice(older)
            opponent_model = _load_opponent(chosen_path)
            if opponent_model is None:
                return sample_action(model, obs, mask)
            return sample_action(opponent_model, obs, mask)
    return _policy


# ---------------------------------------------------------------------------
# Callback for episode-level tracking
# ---------------------------------------------------------------------------

class EpisodeTracker(BaseCallback):
    """Track completed episodes during PPO learn() and handle logging /
    checkpointing at episode boundaries."""

    def __init__(self, start_episode, pbar, model_ref):
        super().__init__(verbose=0)
        self.episode = start_episode
        self.reward_buffer = []
        self.pbar = pbar
        self._model_ref = model_ref  # will be set after model creation

    def _on_step(self) -> bool:
        # Check if any episode ended in the vectorised env
        for idx, done in enumerate(self.locals.get("dones", [])):
            if done:
                ep_reward = self.locals.get("infos", [{}])[idx].get(
                    "episode", {}
                ).get("r", None)
                # Fallback: use the reward from the buffer
                if ep_reward is None:
                    ep_reward = float(self.locals.get("rewards", [0])[idx])
                self.reward_buffer.append(ep_reward)
                self.episode += 1
                self.pbar.update(1)

                # Logging
                if self.episode % LOG_EVERY == 0 and self.reward_buffer:
                    avg_r = np.mean(self.reward_buffer[-LOG_EVERY:])
                    append_reward(self.episode, avg_r)
                    self.pbar.set_postfix(avg_reward=f"{avg_r:.2f}")

                    # Manual entropy coefficient decay
                    progress = self.episode / TOTAL_EPISODES
                    ent_coef = max(ENT_COEF_END, ENT_COEF_START * (1.0 - progress))
                    self.model.ent_coef = ent_coef

                    # Opponent randomisation schedule (high -> low epsilon)
                    opp_epsilon = OPPONENT_EPSILON_END + (
                        OPPONENT_EPSILON_START - OPPONENT_EPSILON_END
                    ) * (1.0 - progress)
                    env = self.model.get_env().envs[0]
                    if hasattr(env, "set_epsilon"):
                        env.set_epsilon(float(opp_epsilon))

                # Checkpoint
                if self.episode % CHECKPOINT_EVERY == 0:
                    save_checkpoint(self.model, self.episode)
                    tqdm.write(
                        f"  💾 Checkpoint saved at episode {self.episode}"
                    )

                    # Refresh self-play policy with league pool
                    pool = get_checkpoint_pool()
                    env = self.model.get_env().envs[0]
                    if hasattr(env, "set_policy"):
                        env.set_policy(
                            make_league_policy_fn(self.model, pool)
                        )

                # Reward graph
                if self.episode % GRAPH_EVERY == 0:
                    save_reward_graph()
                    tqdm.write(
                        f"  📈 Reward graph saved at episode {self.episode}"
                    )

                if self.episode >= TOTAL_EPISODES:
                    return False  # stop training
        return True


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train():
    # Check for fresh start flag
    handle_fresh_start()

    # Resume from checkpoint if available
    ckpt_path, start_episode = latest_checkpoint()

    env = SelfPlayWrapper(WhistEnv(), epsilon=OPPONENT_EPSILON_START)

    if ckpt_path is not None:
        print(f"► Attempting resume from checkpoint: {ckpt_path} (episode {start_episode})")
        try:
            model = RecurrentPPO.load(ckpt_path, env=env, device="cpu")
            # Apply updated schedule to resumed model
            model.learning_rate = LR_SCHEDULE
            model.ent_coef = ENT_COEF_START  # will be decayed by EpisodeTracker
            model._setup_lr_schedule()
            print("  ✓ Recurrent checkpoint loaded")
        except Exception as exc:
            print(f"  ⚠️ Could not load checkpoint with new architecture: {exc}")
            print("  ↳ Starting fresh training run instead.")
            ckpt_path = None
            start_episode = 0
            model = RecurrentPPO(
                "MlpLstmPolicy",
                env,
                learning_rate=LR_SCHEDULE,
                ent_coef=ENT_COEF_START,
                policy_kwargs=POLICY_KWARGS,
                **PPO_KWARGS,
            )
    else:
        print("► Starting fresh training (episode 0)")
        start_episode = 0
        model = RecurrentPPO(
            "MlpLstmPolicy",
            env,
            learning_rate=LR_SCHEDULE,
            ent_coef=ENT_COEF_START,
            policy_kwargs=POLICY_KWARGS,
            **PPO_KWARGS,
        )

    # Wire self-play policy with league pool
    pool = get_checkpoint_pool()
    if pool:
        env.set_policy(make_league_policy_fn(model, pool))
    else:
        env.set_policy(make_policy_fn(model))

    remaining = TOTAL_EPISODES - start_episode
    if remaining <= 0:
        print("Training already complete.")
        return

    pbar = tqdm(total=remaining, desc="Training", unit="ep", file=sys.stdout)
    tracker = EpisodeTracker(start_episode, pbar, model)

    # Estimate total timesteps needed (with margin)
    total_timesteps = remaining * STEPS_PER_EPISODE * 2
    model.learn(
        total_timesteps=total_timesteps,
        callback=tracker,
        reset_num_timesteps=(start_episode == 0),
    )

    pbar.close()

    # Final save
    save_checkpoint(model, tracker.episode)
    save_reward_graph()
    print(f"\nTraining complete. Final episode: {tracker.episode}")


if __name__ == "__main__":
    train()
