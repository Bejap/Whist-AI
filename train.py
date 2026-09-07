"""Training script for Whist RL agent using PPO with self-play."""


print("A: starting module load", flush=True)

import csv
import glob
import os
import random
import re
import shutil
import sys
import time
import subprocess

print("B: stdlib imports done", flush=True)

import numpy as np
print("C: numpy imported", flush=True)

import torch
print("D: torch imported", flush=True)

import matplotlib
print("E: matplotlib imported", flush=True)

matplotlib.use("Agg")
import matplotlib.pyplot as plt
print("F: pyplot imported", flush=True)

from sb3_contrib import MaskablePPO
print("G: stable_baselines3 imported", flush=True)

from stable_baselines3.common.callbacks import BaseCallback
print("H: callbacks imported", flush=True)

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
print("I: SubprocVecEnv imported", flush=True)

from tqdm import tqdm
print("J: tqdm imported", flush=True)

from whist_env import SelfPlayWrapper, WhistEnv
print("K: whist_env imported", flush=True)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CHECKPOINT_DIR = "checkpoints"
REWARDS_CSV = "rewards.csv"
WINRATE_CSV = "winrate.csv"
GRAPH_DIR = "graphs"
TOTAL_EPISODES = 1_000_000
CHECKPOINT_EVERY = 10_000
GRAPH_EVERY = 25_000
LOG_EVERY = 500
EVAL_EVERY = 10_000          # win-rate eval vs frozen baseline
EVAL_EPISODES = 200          # episodes per eval batch
KEEP_CHECKPOINTS = 10
DEVICE = os.getenv("WHIST_DEVICE", "auto")
GPU_UTILIZATION_CAP_PERCENT = int(os.getenv("WHIST_GPU_CAP_PERCENT", "60"))
GPU_UTILIZATION_CHECK_INTERVAL_SECS = float(
    os.getenv("WHIST_GPU_CAP_CHECK_INTERVAL_SECS", "5")
)
GPU_UTILIZATION_POLL_SLEEP_SECS = float(
    os.getenv("WHIST_GPU_CAP_SLEEP_SECS", "0.15")
)

# Heartbeat: print a liveness message every this many seconds even if no
# episodes have finished yet.  Set to 0 to disable.
HEARTBEAT_INTERVAL_SECS = 10

# ---------------------------------------------------------------------------
# Optimization #1 & #2: bigger rollouts + more parallel envs
# ---------------------------------------------------------------------------
NUM_ENVS = 1 if os.name == "nt" else 4
# Windows process-spawning duplicates the torch import across workers and can
# exhaust the paging file; keep Windows in-process by default.
VEC_ENV_CLASS = DummyVecEnv if os.name == "nt" else SubprocVecEnv

# MaskablePPO hyper-parameters – tuned for the current feed-forward baseline
# n_steps per env; total rollout = NUM_ENVS x n_steps = 4 x 2048 = 8192
PPO_KWARGS = dict(
    n_steps=2048,
    batch_size=1024,
    n_epochs=6,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    verbose=0,
    device=DEVICE,
)

# Approximate timesteps per episode (13 tricks, 1 action per trick for the
# learning player in the self-play wrapper).
STEPS_PER_EPISODE = 13

# ---------------------------------------------------------------------------
# Optimization #4 & #5: wider league pool + scheduled latest-policy prob
# ---------------------------------------------------------------------------
LEAGUE_POOL_SIZE = 15        # number of checkpoints to keep in the opponent pool
LEAGUE_LATEST_PROB_START = 0.50   # more varied/weaker opponents early on
LEAGUE_LATEST_PROB_END = 0.80     # mostly self-play once policy matures
OPPONENT_EPSILON_START = 0.20
OPPONENT_EPSILON_END = 0.03

# Larger MLP to utilise GPU compute
POLICY_KWARGS = dict(net_arch=[256, 256, 128])

# Entropy coefficient decay parameters (decayed manually in EpisodeTracker)
ENT_COEF_START = 0.01
ENT_COEF_END = 0.001

# Keep more checkpoints around since the league pool now wants up to 15
KEEP_CHECKPOINTS = max(KEEP_CHECKPOINTS, LEAGUE_POOL_SIZE)

# Frozen baseline checkpoint (for win-rate evaluation). Set once the first
# checkpoint exists; never updated again so win-rate is comparable over time.
BASELINE_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "baseline.pth")


_GPU_CAP_DISABLED = False


def get_gpu_utilization_percent():
    """Return the current GPU utilization percent, or None if unavailable."""
    global _GPU_CAP_DISABLED
    if _GPU_CAP_DISABLED or not torch.cuda.is_available():
        return None

    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        _GPU_CAP_DISABLED = True
        return None

    first_line = output.strip().splitlines()[:1]
    if not first_line:
        return None

    try:
        return float(first_line[0])
    except ValueError:
        return None


def maybe_throttle_gpu():
    """Briefly yield training when the GPU is above the requested cap."""
    if GPU_UTILIZATION_CAP_PERCENT <= 0:
        return

    util = get_gpu_utilization_percent()
    if util is None:
        return

    if util <= GPU_UTILIZATION_CAP_PERCENT:
        return

    over = util - GPU_UTILIZATION_CAP_PERCENT
    sleep_secs = GPU_UTILIZATION_POLL_SLEEP_SECS * max(1.0, over / 5.0)
    time.sleep(min(sleep_secs, 1.0))


# ---------------------------------------------------------------------------
# Schedules
# ---------------------------------------------------------------------------

def linear_schedule(start: float, end: float):
    """Return a callable that linearly decays from start to end.

    The callable receives `progress_remaining` (1.0 -> 0.0) from SB3.
    """
    def _schedule(progress_remaining: float) -> float:
        return end + (start - end) * progress_remaining
    return _schedule


def linear_value(start: float, end: float, progress: float) -> float:
    """Linearly interpolate from start to end as progress goes 0 -> 1."""
    progress = min(max(progress, 0.0), 1.0)
    return start + (end - start) * progress


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
    # Delete rewards.csv / winrate.csv
    if os.path.exists(REWARDS_CSV):
        os.remove(REWARDS_CSV)
    if os.path.exists(WINRATE_CSV):
        os.remove(WINRATE_CSV)
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


def ensure_baseline_checkpoint(model, episode):
    """Save a one-time frozen baseline checkpoint for win-rate evaluation.

    This is saved once (the earliest available checkpoint) and never
    overwritten, so win-rate-vs-baseline is comparable across all of
    training.
    """
    if os.path.exists(BASELINE_CHECKPOINT):
        return
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    model.save(BASELINE_CHECKPOINT)
    print(f"  ★ Frozen baseline checkpoint saved at episode {episode}")


def append_reward(episode, reward):
    """Append a row to rewards.csv."""
    write_header = not os.path.exists(REWARDS_CSV)
    with open(REWARDS_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["episode", "avg_reward"])
        writer.writerow([episode, f"{reward:.4f}"])


def append_winrate(episode, win_rate):
    """Append a row to winrate.csv."""
    write_header = not os.path.exists(WINRATE_CSV)
    with open(WINRATE_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["episode", "win_rate_vs_baseline"])
        writer.writerow([episode, f"{win_rate:.4f}"])


def save_reward_graph():
    """Read rewards.csv and winrate.csv and save graphs to graphs/."""
    os.makedirs(GRAPH_DIR, exist_ok=True)

    # --- Reward graph -----------------------------------------------------
    if os.path.exists(REWARDS_CSV):
        episodes, rewards = [], []
        with open(REWARDS_CSV, newline="") as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header
            for row in reader:
                if len(row) >= 2:
                    episodes.append(int(row[0]))
                    rewards.append(float(row[1]))

        if episodes:
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

    # --- Win-rate vs frozen baseline graph --------------------------------
    if os.path.exists(WINRATE_CSV):
        episodes, win_rates = [], []
        with open(WINRATE_CSV, newline="") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if len(row) >= 2:
                    episodes.append(int(row[0]))
                    win_rates.append(float(row[1]))

        if episodes:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(episodes, win_rates, marker="o", linewidth=1.5,
                    color="green", label="win rate vs frozen baseline")
            ax.axhline(0.5, color="gray", linestyle="--", linewidth=1,
                       label="50% (parity)")
            ax.set_xlabel("Episode")
            ax.set_ylabel("Win Rate")
            ax.set_ylim(0.0, 1.0)
            ax.set_title(f"Win Rate vs Frozen Baseline (up to episode {episodes[-1]})")
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            path = os.path.join(GRAPH_DIR, f"winrate_ep_{episodes[-1]}.png")
            fig.savefig(path, dpi=100)
            plt.close(fig)


def sample_action(model, obs, mask):
    """Sample a masked action from the model using logits-based sampling."""
    obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(model.device)
    with torch.no_grad():
        logits = model.policy.get_distribution(obs_t).distribution.logits
    logits = logits.squeeze(0).cpu().numpy()
    logits[mask == 0] = -1e8
    probs = np.exp(logits - logits.max())
    probs = probs / probs.sum()
    return int(np.random.choice(len(probs), p=probs))


def greedy_action(model, obs, mask):
    """Pick the highest-probability valid action (used for evaluation)."""
    obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(model.device)
    with torch.no_grad():
        logits = model.policy.get_distribution(obs_t).distribution.logits
    logits = logits.squeeze(0).cpu().numpy()
    logits[mask == 0] = -1e8
    return int(np.argmax(logits))


def make_policy_fn(model):
    """Create a policy function for self-play opponents."""
    def _policy(obs, mask):
        return sample_action(model, obs, mask)
    return _policy


def make_league_policy_fn(model, pool_paths, latest_prob):
    """Create a league-based policy function.

    - `latest_prob` chance: use the latest (live) model
    - otherwise: use a randomly chosen older checkpoint from the pool

    Older checkpoints are loaded lazily and cached to avoid repeated I/O.
    """
    # Cache for loaded opponent policies (path -> model)
    _cache = {}

    def _load_opponent(path):
        if path not in _cache:
            try:
                # Load policy parameters only (no env needed for inference)
                _cache[path] = MaskablePPO.load(path, device=DEVICE)
            except Exception as exc:
                print(f"  ⚠️ Skipping incompatible league checkpoint {path}: {exc}")
                _cache[path] = None
        return _cache[path]

    def _policy(obs, mask):
        # League selection
        if len(pool_paths) <= 1:
            return sample_action(model, obs, mask)
        use_latest = random.random() < latest_prob
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
# Optimization #3: win-rate vs frozen baseline evaluation
# ---------------------------------------------------------------------------

def evaluate_win_rate(model, n_episodes=EVAL_EPISODES):
    """Play `n_episodes` of the learning model (greedy) vs the frozen
    baseline checkpoint (sampled) and return the fraction of episodes the
    learning player's team wins.

    Runs in a separate single-process env so it doesn't disturb the
    training SubprocVecEnv. Cheap relative to training (EVAL_EPISODES is
    small and greedy/sampled inference is fast on CPU/GPU).
    """
    if not os.path.exists(BASELINE_CHECKPOINT):
        return None

    try:
        baseline_model = MaskablePPO.load(BASELINE_CHECKPOINT, device=DEVICE)
    except Exception as exc:
        print(f"  ⚠️ Could not load baseline for eval: {exc}")
        return None

    env = WhistEnv()
    wins = 0

    for _ in range(n_episodes):
        obs, info = env.reset()
        learning_player = env.current_player
        team = TEAMS_FOR_EVAL[learning_player]
        done = False

        while not done:
            mask = env.action_mask()
            if env.current_player == learning_player:
                action = greedy_action(model, obs, mask)
            else:
                action = sample_action(baseline_model, obs, mask)
            obs, _r, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            obs = env._get_obs()  # refresh for the *new* current_player

        if env.team_tricks[team] > env.team_tricks[1 - team]:
            wins += 1

    return wins / n_episodes


# Local copy of the team mapping (avoid importing private module internals)
TEAMS_FOR_EVAL = {0: 0, 1: 1, 2: 0, 3: 1}


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
        self._t0 = time.monotonic()
        self._last_hb = self._t0
        self._last_gpu_check = 0.0

    def _on_training_start(self) -> None:
        tqdm.write("► First callback activity observed — training loop is running.")

    def _on_step(self) -> bool:
        now = time.monotonic()
        if now - self._last_gpu_check >= GPU_UTILIZATION_CHECK_INTERVAL_SECS:
            maybe_throttle_gpu()
            self._last_gpu_check = now

        # Heartbeat: emit a liveness line on a wall-clock interval
        if HEARTBEAT_INTERVAL_SECS > 0:
            now = time.monotonic()
            if now - self._last_hb >= HEARTBEAT_INTERVAL_SECS:
                elapsed = now - self._t0
                fps = self.num_timesteps / elapsed if elapsed > 0 else 0.0
                tqdm.write(
                    f"  ♥ heartbeat | steps={self.num_timesteps:,}"
                    f" | episodes={self.episode:,}"
                    f" | fps={fps:.0f}"
                    f" | elapsed={elapsed:.0f}s"
                )
                self._last_hb = now

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
                    try:
                        self.model.get_env().env_method("set_epsilon", float(opp_epsilon))
                    except Exception as exc:
                        tqdm.write(f"  ⚠️ set_epsilon failed: {exc}")

                # Checkpoint
                if self.episode % CHECKPOINT_EVERY == 0:
                    save_checkpoint(self.model, self.episode)
                    tqdm.write(
                        f"  💾 Checkpoint saved at episode {self.episode}"
                    )

                    # One-time frozen baseline for win-rate evaluation
                    ensure_baseline_checkpoint(self.model, self.episode)

                    # Refresh self-play policy with league pool, using the
                    # scheduled latest-policy probability.
                    pool = get_checkpoint_pool()
                    progress = self.episode / TOTAL_EPISODES
                    latest_prob = linear_value(
                        LEAGUE_LATEST_PROB_START, LEAGUE_LATEST_PROB_END, progress
                    )
                    try:
                        self.model.get_env().env_method(
                            "set_policy",
                            make_league_policy_fn(self.model, pool, latest_prob),
                        )
                    except Exception as exc:
                        tqdm.write(f"  ⚠️ set_policy failed: {exc}")

                # Win-rate evaluation vs frozen baseline
                if self.episode % EVAL_EVERY == 0:
                    win_rate = evaluate_win_rate(self.model)
                    if win_rate is not None:
                        append_winrate(self.episode, win_rate)
                        tqdm.write(
                            f"  🏆 Win rate vs frozen baseline at episode "
                            f"{self.episode}: {win_rate:.1%}"
                        )

                # Reward graph
                if self.episode % GRAPH_EVERY == 0:
                    save_reward_graph()
                    tqdm.write(
                        f"  📈 Reward/win-rate graphs saved at episode {self.episode}"
                    )

                if self.episode >= TOTAL_EPISODES:
                    return False  # stop training
        return True


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train():
    print("► Training run started.", flush=True)

    # Check for fresh start flag
    handle_fresh_start()

    # Resume from checkpoint if available
    ckpt_path, start_episode = latest_checkpoint()
    if ckpt_path is not None:
        print(
            f"► Checkpoint status: found {ckpt_path} (episode {start_episode})",
            flush=True,
        )
    else:
        print("► Checkpoint status: no checkpoint found (fresh start).", flush=True)

    def make_env():
        def _init():
            return SelfPlayWrapper(WhistEnv(), epsilon=OPPONENT_EPSILON_START)
        return _init

    print(f"► Creating {VEC_ENV_CLASS.__name__} with {NUM_ENVS} workers...", flush=True)
    env_init_t0 = time.monotonic()
    print("train() entered", flush=True)
    print("about to create vec env", flush=True)
    env = VEC_ENV_CLASS([make_env() for _ in range(NUM_ENVS)])
    print("vec env created", flush=True)
    print(
        f"  ✓ SubprocVecEnv ready in {time.monotonic() - env_init_t0:.1f}s",
        flush=True,
    )

    if ckpt_path is not None:
        print(f"► Attempting resume from checkpoint: {ckpt_path} (episode {start_episode})")
        try:
            model = MaskablePPO.load(ckpt_path, env=env, device=DEVICE)
            # Apply updated schedule to resumed model
            model.learning_rate = LR_SCHEDULE
            model.ent_coef = ENT_COEF_START  # will be decayed by EpisodeTracker
            model._setup_lr_schedule()
            print("  ✓ PPO checkpoint loaded")
        except Exception as exc:
            print(f"  ⚠️ Could not load checkpoint: {exc}")
            print("  ↳ Starting a fresh PPO run.")
            ckpt_path = None
            start_episode = 0
            model = MaskablePPO(
                "MlpPolicy",
                env,
                learning_rate=LR_SCHEDULE,
                ent_coef=ENT_COEF_START,
                policy_kwargs=POLICY_KWARGS,
                **PPO_KWARGS,
            )
    else:
        print("► Starting fresh training (episode 0)")
        start_episode = 0
        model = MaskablePPO(
            "MlpPolicy",
            env,
            learning_rate=LR_SCHEDULE,
            ent_coef=ENT_COEF_START,
            policy_kwargs=POLICY_KWARGS,
            **PPO_KWARGS,
        )

    print(f"► Requested device: {DEVICE}", flush=True)
    print(f"► Active device: {getattr(model, 'device', 'unknown')}", flush=True)
    print(f"► GPU usage cap: {GPU_UTILIZATION_CAP_PERCENT}% soft limit", flush=True)
    print(
        f"► Parallel envs: {NUM_ENVS} (total rollout: {NUM_ENVS * PPO_KWARGS['n_steps']} steps)",
        flush=True,
    )

    # Wire self-play policy with league pool
    pool = get_checkpoint_pool()
    progress0 = start_episode / TOTAL_EPISODES
    latest_prob0 = linear_value(LEAGUE_LATEST_PROB_START, LEAGUE_LATEST_PROB_END, progress0)
    policy_fn = (
        make_league_policy_fn(model, pool, latest_prob0)
        if pool
        else make_policy_fn(model)
    )
    try:
        env.env_method("set_policy", policy_fn)
    except Exception as exc:
        print(f"  ⚠️ Initial set_policy failed: {exc}")

    remaining = TOTAL_EPISODES - start_episode
    if remaining <= 0:
        print("Training already complete.")
        env.close()
        return

    pbar = tqdm(total=remaining, desc="Training", unit="ep", file=sys.stdout)
    tracker = EpisodeTracker(start_episode, pbar, model)

    # Estimate total timesteps needed (with margin)
    total_timesteps = remaining * STEPS_PER_EPISODE
    print(f"► Total timesteps planned: {total_timesteps:,}", flush=True)
    print(
        f"► Starting model.learn() — heartbeat every {HEARTBEAT_INTERVAL_SECS}s …",
        flush=True,
    )
    model.learn(
        total_timesteps=total_timesteps,
        callback=tracker,
        reset_num_timesteps=(start_episode == 0),
    )

    pbar.close()
    env.close()

    # Final save
    save_checkpoint(model, tracker.episode)
    save_reward_graph()
    print(f"\nTraining complete. Final episode: {tracker.episode}")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    train()