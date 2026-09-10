# Whist-AI
Creating an AI to play whist better than humans

## Setup

```bash
pip install -r requirements.txt
```

## Published models

A small portable model registry is stored in `published_models/` with Git LFS.
Install Git LFS before cloning or pulling the model files:

```powershell
git lfs install
git lfs pull
```

See `published_models/MODEL_REGISTRY.md` for the available checkpoints. The
full training directories remain local-only so the repository stays usable.

## Training

Start a fresh training run (1,000,000 episodes by default):

```bash
python train.py
```

Checkpoints are saved every 10,000 episodes to the `checkpoints/` directory,
a reward log is written to `rewards.csv`, and a reward graph is saved to
`graphs/` every 25,000 episodes.

Benchmark a checkpoint against fixed opponents and write results to
`benchmarks.csv`:

```bash
python evaluate.py --episodes 1000 --opponent all --mcts-sims 0
```

Generate the combined training and benchmark dashboard:

```bash
python plot.py
```

This also preserves the original standalone graph pair. The original run is
written to `graphs/reward_ep_*.png` and `graphs/winrate_ep_*.png`; the
multi-seat run gets the same pair in `graphs_multiseat/`, alongside the newer
dashboard graphics.

The dashboard includes complete episode return, win rate versus the frozen
baseline, fixed-opponent win rates with confidence bands, and average trick
difference. Per-game results are also written to `benchmark_games.csv` for
seat/team and trump-condition breakdowns. Benchmark results are independent
of the changing self-play pool.

### Resuming from a checkpoint

Training **automatically resumes** from the latest checkpoint found in
`checkpoints/`.  Simply run `python train.py` again and it will pick up
where it left off.

To resume from a **specific** checkpoint (e.g. after copying one from
another machine), make sure the desired `.pth` file is in the `checkpoints/`
directory and remove any newer checkpoint files so that it becomes the
latest one, then run `python train.py`.

```bash
# Example: resume from episode 5000
ls checkpoints/          # verify whist_cp_5000.pth exists
python train.py          # automatically loads the latest checkpoint
```

You can also adjust `TOTAL_EPISODES` in `train.py` to extend or shorten the
default 1,000,000-episode run.

### Configuration

Key parameters in `train.py`:

| Parameter | Default | Description |
|---|---|---|
| `TOTAL_EPISODES` | 1,000,000 | Total training episodes |
| `CHECKPOINT_EVERY` | 10,000 | Save a checkpoint every N episodes |
| `GRAPH_EVERY` | 25,000 | Save a reward graph every N episodes |
| `KEEP_CHECKPOINTS` | 10 | Number of recent checkpoints to keep |
| `LOG_EVERY` | 500 | Log average reward every N episodes |

Training uses **MaskablePPO** so the policy only samples cards that are legal
under Whist's follow-suit rule. Opponent self-play uses an epsilon curriculum
that starts more random and becomes stronger over time. The current baseline
policy is feed-forward; `models.py` contains an optional transformer extractor
that is not enabled by default.

### Multi-seat experiment

The stronger shared-policy experiment rotates the learning seat across all
four players while retaining the latest-policy league opponents. It writes to
separate files and does not overwrite the original run:

```powershell
$env:WHIST_TOTAL_EPISODES="1000000"
python train_multiseat.py
```

Use a short validation run first:

```powershell
$env:WHIST_TOTAL_EPISODES="1000"
$env:WHIST_CHECKPOINT_EVERY="500"
$env:WHIST_EVAL_EVERY="500"
$env:WHIST_EVAL_EPISODES="100"
python train_multiseat.py
```

### Strong transformer experiment

`train_strong.py` is an isolated experiment for stronger strategic play. It
uses the transformer card extractor, lower heuristic shaping, a larger terminal
team-result reward, and promotion by fixed rule-opponent performance. Each
checkpoint receives a fast raw-policy evaluation over balanced team assignments
and a separate hidden-hand MCTS evaluation. The best raw-policy checkpoint is
saved as `checkpoints_strong/best_rule.pth`.

```powershell
python train_strong.py
```

It writes checkpoints, reward/win-rate graphs, and benchmark scores to its own
`*_strong` paths and never replaces the original or multi-seat runs. By
default it saves a checkpoint every 5,000 episodes, a graph every 10,000
episodes, and logs reward every 250 episodes. Each checkpoint uses a compact
50-game raw benchmark plus a 4-game hidden-hand MCTS spot check.

### Fast competitive experiment

For a faster strategic experiment, use `train_competitive.py`. It keeps the
MLP policy and four-seat training, but mixes a real fixed rule player into
35% of opponent turns. It evaluates 200 complete balanced games at every
checkpoint and promotes the best rule-player result to
`checkpoints_competitive/best_rule.pth`.

```powershell
python train_competitive.py
```

Its outputs are isolated under `checkpoints_competitive/`,
`rewards_competitive.csv`, `graphs_competitive/`, and
`competitive_checkpoint_scores.csv`.

### Post-competitive self-play

After competitive training is complete, start the next phase with:

```powershell
python train_post_selfplay.py
```

It seeds an isolated run from `checkpoints_competitive/best_rule.pth` and
mixes current/historical policies, the rule player (20%), and random legal
play (10%). It does not modify competitive checkpoints. Results go to
`checkpoints_post_selfplay/`, `graphs_post_selfplay/`, and
`post_selfplay_checkpoint_scores.csv`.

### Anchor self-play experiment

The more controlled follow-up keeps the best competitive model as a fixed
anchor. It mixes 45% anchor policy, 25% league policies, 20% rule-player
turns, and 10% random legal turns. Every checkpoint is evaluated against both
the rule player and the fixed anchor:

```powershell
python train_anchor_selfplay.py
```

It starts from `checkpoints_competitive/best_rule.pth` and writes only to
`checkpoints_anchor_selfplay/`, `graphs_anchor_selfplay/`, and
`anchor_selfplay_checkpoint_scores.csv`.

### GPU usage

By default, training/inference use `WHIST_DEVICE=auto`, which lets SB3/PyTorch
pick GPU when available and fall back to CPU otherwise.

Training also applies a soft GPU utilization cap of 60% by default. When
`nvidia-smi` reports the GPU above that threshold, the training loop briefly
yields so the card has more headroom for other work.

```bash
# Auto-select (default)
python train.py

# Force a specific device
WHIST_DEVICE=cuda python train.py
WHIST_DEVICE=cpu python train.py

# Adjust the soft GPU cap if needed
WHIST_GPU_CAP_PERCENT=60 python train.py
WHIST_GPU_CAP_PERCENT=0 python train.py
```

You can use the same variable for gameplay:

```bash
WHIST_DEVICE=auto python play.py --mode watch
```

If it still runs on CPU, your PyTorch install likely has no GPU backend.
NVIDIA requires a CUDA-enabled PyTorch build; AMD requires a ROCm-enabled
PyTorch build (Linux) or another supported backend. On Windows/macOS, AMD GPU
support in PyTorch is limited and CPU fallback is common.

## Playing

### Browser table

Launch the interactive card table:

```bash
python web_app.py
```

Then open `http://127.0.0.1:8000` in a browser. Click a legal card in your
hand to play it; the AI completes the other seats automatically. The browser
table uses the latest checkpoint and 16 MCTS simulations by default. Set
`WHIST_WEB_MCTS_SIMS=64` for stronger but slower decisions.

Watch the trained agent play a full round:

```bash
python play.py
```

Use MCTS-guided inference (enabled by default with 64 simulations per move):

```bash
python play.py --mode watch --mcts-sims 64
```

Save a game for later analysis as JSON:

```bash
python play.py --mode play --replay replays/my_game.json
```

The replay records every action, the legal cards at that point, rewards, and
the score after each action.

## Environment

The Whist environment (`whist_env.py`) follows the Gymnasium API.

**Observation space** — 340-dimensional vector:
- Own hand (52 bits)
- Per-player played cards (4 × 52 bits)
- Current trick cards (52 bits)
- Trump suit (5 bits, one-hot)
- Team tricks (2 floats, normalised)
- Learning player seat (4 bits, one-hot)
- Current trick winner (4 bits, one-hot)
- Lead suit (5 bits, one-hot; includes "no lead yet")
- Trump exhaustion flags (4 bits, one per seat)
- Trick position in current trick (4 bits, one-hot)

**Action space** — Discrete(52), masked to valid cards in hand.

**Reward shaping:**
- +2 / −2 per trick won / lost
- +0.3 when an opponent Ace is captured in a trick won by your team
- +0.3 when an opponent King is captured in a trick won by your team
- +0.2 when an opponent Queen is captured in a trick won by your team
- +0.3 bonus for winning a trick with a trump card
- +0.2 bonus for winning with the highest lead-suit card
- −0.1 penalty for wasting a trump on a trick already won by your team
- −0.5 penalty for selecting an invalid action
- ±2.0 terminal bonus / penalty for winning / losing the round
