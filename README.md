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

## Playing

Watch the trained agent play a full round:

```bash
python play.py
```

## Environment

The Whist environment (`whist_env.py`) follows the Gymnasium API.

**Observation space** — 167-dimensional vector:
- Own hand (52 bits)
- Played cards (52 bits)
- Current trick cards (52 bits)
- Trump suit (5 bits, one-hot)
- Team tricks (2 floats, normalised)
- Learning player seat (4 bits, one-hot)

**Action space** — Discrete(52), masked to valid cards in hand.

**Reward shaping:**
- +1 / −1 per trick won / lost
- +0.3 bonus for winning a trick with a trump card
- +0.2 bonus for winning with the highest lead-suit card
- −0.1 penalty for wasting a trump on a trick already won by your team
- −0.5 penalty for selecting an invalid action
- ±3.0 terminal bonus / penalty for winning / losing the round
