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

Training now uses standard **PPO** with the default **MlpPolicy** network
for faster CPU training.
Opponent self-play also uses an epsilon curriculum that starts more random and
becomes stronger over time.

## Playing

Watch the trained agent play a full round:

```bash
python play.py
```

### Graphical interface (GUI)

Play interactively against three AI opponents in a windowed card table:

```bash
python gui.py
```

- You are **P1** (bottom, cards face-up and clickable with the mouse).
- AI players are **P2**, **P3**, and **P4**.
- **Team 0**: You (P1) & P3 — **Team 1**: P2 & P4.
- If no checkpoint is found the AI plays randomly.
- The window is resizable; target resolution is 1024 × 768.

For future GPU-focused upgrades (RecurrentPPO, Transformer extractor, MCTS),
see `Improvements.md`.

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
- +1 / −1 per trick won / lost
- +0.4 bonus for winning a trick with a trump card
- +0.2 bonus for winning with the highest lead-suit card
- −0.1 penalty for wasting a trump on a trick already won by your team
- −0.5 penalty for selecting an invalid action
- ±3.0 terminal bonus / penalty for winning / losing the round
