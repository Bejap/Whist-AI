# Whist-AI
Creating an AI to play whist better than humans

## Setup

```bash
pip install -r requirements.txt
```

## Training

Start a fresh training run (100,000 episodes by default):

```bash
python train.py
```

Checkpoints are saved every 10,000 episodes to the `checkpoints/` directory,
a reward log is written to `rewards.csv`, and a reward graph is saved to
`graphs/` every 25,000 episodes.

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

You can also adjust `TOTAL_EPISODES` in `train.py` to extend training
beyond the default 100,000 episodes.

### Configuration

Key parameters in `train.py`:

| Parameter | Default | Description |
|---|---|---|
| `TOTAL_EPISODES` | 100,000 | Total training episodes |
| `CHECKPOINT_EVERY` | 10,000 | Save a checkpoint every N episodes |
| `GRAPH_EVERY` | 25,000 | Save a reward graph every N episodes |
| `KEEP_CHECKPOINTS` | 10 | Number of recent checkpoints to keep |
| `LOG_EVERY` | 500 | Log average reward every N episodes |

Training now uses a **RecurrentPPO (LSTM)** policy with a custom
**Transformer-based card feature extractor**.
Opponent self-play also uses an epsilon curriculum that starts more random and
becomes stronger over time.

### GPU usage

By default, training/inference use `WHIST_DEVICE=cpu` so SB3 trains on CPU,
which is the correct choice for MlpPolicy.  Set `WHIST_DEVICE=cuda` to train
on NVIDIA GPU if you have a CUDA-enabled PyTorch build.

```bash
# Auto-select (default)
python train.py

# Force a specific device
WHIST_DEVICE=cuda python train.py
WHIST_DEVICE=cpu python train.py
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

Watch the trained agent play a full round:

```bash
python play.py
```

Use MCTS-guided inference (enabled by default with 64 simulations per move):

```bash
python play.py --mode watch --mcts-sims 64
```

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
