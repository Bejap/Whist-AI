# Whist-AI
Creating an AI to play whist better than humans

Esmakker training runs can be isolated with `--run-name`. For example,
`--run-name esmakker_v2` writes checkpoints under
`checkpoints/esmakker_v2/` and reports under `graphs/esmakker_v2/` without
overwriting another run.

## Esmakker card-play curriculum

Step 1 trains card play under a fixed numeric contract, without bidding. The
learner is always the declarer, receives a random trump and partner suit, and
must learn legal trick-taking decisions before bidding is introduced. The
separate trainer is `training/cardplay/train_esmakker_cardplay.py`; its default output directory
is `checkpoints/esmakker_cardplay_v1/` with metrics in
`graphs/esmakker_cardplay_v1/cardplay_metrics.csv`.

Start a fresh contract-7 curriculum run with:

```powershell
python -u -m training.cardplay.train_esmakker_cardplay `
	--run-name esmakker_cardplay_v1 `
	--contract 7 `
	--timesteps 500000 `
	--device cuda `
	--gpu-memory-fraction 0.30 `
	--n-steps 2048 `
	--batch-size 512
```

Benchmark a trained specialist deterministically across all numeric contracts
with fixed seeds:

```powershell
python -m training.cardplay.evaluate_cardplay `
	--checkpoint checkpoints/esmakker_cardplay_v2/latest.zip `
	--contract all `
	--episodes 500 `
	--device cuda
```

The benchmark writes per-game results to
`graphs/esmakker_cardplay_v2/cardplay_benchmark_games.csv` and aggregate
contract results to `graphs/esmakker_cardplay_v2/cardplay_benchmark_summary.csv`.
In these files, `success` means the learner received positive settlement;
`contract_success` records whether the declarer's contract succeeded. This
distinction matters when the learner is defending.
Use the same `--seed` and episode count when comparing checkpoints.

To compare opponent tables, run all three modes:

```powershell
python -m training.cardplay.evaluate_cardplay `
	--checkpoint checkpoints/esmakker_cardplay_v2/latest.zip `
	--opponents all `
	--opponent-checkpoint checkpoints/esmakker_cardplay_v2/latest.zip `
	--contract all `
	--episodes 500 `
	--device cuda
```

The tables are `random`, `rules`, and `learned`. Learned opponents must use a
checkpoint with the same 422-feature observation and 52-action space. For a
genuine historical-opponent comparison, preserve an older v2 checkpoint
under a separate filename before continuing training; `latest.zip` is a
rolling file and is not a historical archive.

Use `--resume` with the same run name to continue it. The current bidding run
under `esmakker_v2` is not modified by this curriculum.

The upgraded specialist can train across numeric contracts 7 through 13 and
tracks played cards, which player played each card, and inferred player/suit
voids. Because this changes
the observation and policy architecture, start it under a new run name rather
than resuming the older contract-7 checkpoint:

With `--contract all`, numeric contracts are sampled using Zipf weighting in
the order `9, 8, 7, 10, 11, 12, 13`, with weights `1, 1/2, 1/3, ... 1/7`.

```powershell
python -u -m training.cardplay.train_esmakker_cardplay `
	--run-name esmakker_cardplay_v2 `
	--contract all `
	--timesteps 1000000 `
	--device cuda `
	--gpu-memory-fraction 0.30 `
	--n-steps 2048 `
	--batch-size 512
```

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

## Esmakker Whist

### Training

The Esmakker rules engine has a separate masked PPO adapter and trainer. It
trains one randomized seat against three opponents drawn from a frozen-policy
league. The same historical policy controls all three opponents for a round,
including bidding, trump, partner-suit and card-play decisions. A legal
rule-based opponent is used only while no compatible snapshot exists.

```powershell
python -m training.esmakker.train_esmakker --timesteps 250000
```

To train with CUDA while limiting this process to 30% of the GPU VRAM, keep
league opponents on the CPU to reduce GPU contention:

```powershell
python -m training.esmakker.train_esmakker --timesteps 250000 --device cuda --opponent-device cpu --gpu-memory-fraction 0.30
```

The memory fraction limits VRAM reserved by this process; it does not impose a
hard 30% GPU-compute-utilization limit. Use `--opponent-device cuda` only when
you want frozen league opponents on the GPU as well. The cap can be changed,
for example, with `--gpu-memory-fraction 0.20`.

Resume the latest Esmakker checkpoint with:

```powershell
python -m training.esmakker.train_esmakker --resume --timesteps 250000
```

Checkpoints are saved to `checkpoints/esmakker/latest.zip`.

Paskrig adds a contract-state feature to the policy observation. Esmakker
checkpoints created before Paskrig are incompatible; start a fresh Esmakker
training run after deleting `checkpoints/esmakker/` and `graphs/esmakker/`.

Self-play snapshots are saved every 50,000 timesteps under
`checkpoints/esmakker/league/`; the newest 10 are retained. A snapshot is
selected once per round so opponents remain consistent during that auction
and hand. The live learner is never used directly as its own opponent. This
mix of older policies prevents the auction from adapting only to the latest
policy's current weaknesses. The defaults can be changed with
`--snapshot-every`, `--league-size`, and `--opponent-device`.

Self-play is not used as the quality score. A separate benchmark runs every
5,000 training timesteps: 100 games against fixed rule opponents and 100
games with the checkpoint choosing every seat's bids from that seat's own
hand while rule opponents handle declaration and card play. Results are written to
`graphs/esmakker/rule_benchmark.csv` and plotted in
`graphs/esmakker/rule_benchmark_progress.png`. The plot shows both the
overall fixed-rule result and separate lines for learner seats 1 through 4.
The solid black line is the result to judge: above zero settlement means the
checkpoint wins money on average against the same fixed opponents; below zero
means it loses money. The dashed purple line is the model-led-auction
diagnostic, not the primary quality score. Since the learning seat is
randomized, the overall line is the primary comparison; seat lines only check
for seat bias. Incomplete benchmark games are recorded but excluded from
settlement averages, with a 200-decision safety limit.

During training, completed rounds are recorded in
`graphs/esmakker/training_metrics.csv`, and the latest-100-round progress
graph is refreshed at `graphs/esmakker/training_progress.png`. The graph's
top panel shows settlement/payment and the bottom panel shows contract
success. A rising smoothed line is good; compare runs only after the same
number of completed rounds.

Every learning-player bid is also recorded in
`graphs/esmakker/bidding_decisions.csv`. It includes the bid, the bid already
on the table, high-card count, ace count, longest suit, final contract,
contract success, tricks and settlement. This makes overbidding visible: look
for high bids with `success=0` and negative `settlement`, especially when the
hand has few high cards. For a quick PowerShell view:

```powershell
Import-Csv graphs/esmakker/bidding_decisions.csv |
	Where-Object { $_.action -notin @('pass', '') -and $_.success -eq '0' } |
	Select-Object -Last 20
```

For a direct view of the contracts the learning model actually won, open
`graphs/esmakker/bid_distribution.png`. It counts one final contract per
completed round where the randomized learning seat was the declarer, so an
opponent-won auction and an intermediate bid later overbid do not appear as
the model's conclusion. The matching `graphs/esmakker/bid_summary.csv`
contains the count and percentage for each concluded model contract. Use
`graphs/esmakker/bidding_decisions.csv` when you need the separate raw action
distribution, including passes and intermediate bids.

For contract quality, open `graphs/esmakker/contract_success_rate.png` or
`graphs/esmakker/contract_success_summary.csv`. These show, for each contract
won by the learning seat, the auctions won, successful completions, success
rate, average payment, and total payment.

For outcomes per learner action, use
`graphs/esmakker/bid_outcomes.csv`. It reports decisions, completed rounds,
positive-settlement rate, contract-success rate, and average settlement for
each action. The raw `bidding_decisions.csv` contains every learner decision
in an auction, not each opponent bid. Its `final_contract` column is the
winning bid after the complete bidding war.

For a human-readable version, open
`graphs/esmakker/bid_outcomes_report.txt`. It explains each action in plain
language and makes clear that the settlement belongs to the complete round
containing that action.

The policy can choose every legal bid, including `7` through `13`, `Sol`,
`Ren sol`, `Bordlaegger`, and pass. Its observation includes its hand and the
current bid. The bid is therefore learned as a decision under uncertainty:
the policy is rewarded or penalized by the final contract settlement. It is
not given a hand-written meaning for a bid, and it does not learn the Danish
word itself; it learns that a bid commits it to a trick target or nolo limit

For numeric and nolo contracts, terminal training reward is assigned only when
the learning seat is the declarer. This prevents bidding Bordlaegger merely to
provoke an opponent into overbidding and then collecting the opponent's failed
contract payout. Paskrig remains seat-based because it has no declarer.

During opponent bidding, an existing contract is counter-bid with probability
`ESMAKKER_OPPONENT_COUNTER_BID_PROBABILITY` (default `0.15`); otherwise the
opponent passes. Opening bids remain policy-driven. This keeps most auctions
from escalating through unrealistic high contracts while preserving occasional
competitive bidding. Set the variable before launching training to change it.
with a corresponding risk and reward. The implemented meanings are listed in
`esmakker_rules.md`.

For numeric contracts, the terminal reward is based on the actual settlement.
Taking more tricks than the bid does not receive an additional punishment.

An opening pass, when no bid is yet on the table, receives a small immediate
`-0.05` reward penalty by default. This prevents an all-pass auction from
being entirely risk-free without forcing the model to overbid. Passing after
another bid is reward-neutral. Set `ESMAKKER_OPENING_PASS_PENALTY` to tune the
opening-pass value.

Terminal settlement rewards use the actual payment scaled by 50:
`payment / 50`. This keeps rewards manageable while preserving the difference
between a near miss and a larger loss. A successful 13 pays 25 but keeps a
normalized training reward of `+1.0`, a successful
Bordlægger is `+0.48`, and a successful Sol is `+0.12`. Set
`ESMAKKER_SETTLEMENT_REWARD_SCALE` to tune the scale.

The fixed-partnership Whist training environments are retained as experiments.
The intended long-term game is Esmakker Whist, implemented separately in
`esmakker_env.py`. It models bidding, declarer-selected trump, the secret
partner ace or king, nolo contracts, dealer rotation, and settlement. See
`esmakker_rules.md` for the exact implemented rules.

### Esmakker browser demo

Run the first interactive Esmakker table with:

```powershell
python esmakker_web.py
```

Open `http://127.0.0.1:8001`. This prototype makes you declarer on a 9
contract, lets you choose trump and partner suit, and lets you play cards by
clicking them. The other seats currently use random legal play; actions are
kept in the in-memory game history as the foundation for human-game logging
and later imitation learning.

Completed human rounds are appended to
`graphs/benchmarks/esmakker_human_games.jsonl`. For a useful first imitation
dataset, aim for **50-100 complete rounds**. For a robust first model of your
style, **200-500 rounds** is a better target, especially because bidding,
contract choice, trump choice, partner choice, and card play all need examples.

## Training

Start a fresh training run (1,000,000 episodes by default):

```bash
python -m training.selfplay.train
```

Checkpoints are saved every 10,000 episodes to the `checkpoints/` directory,
a reward log is written to `rewards.csv`, and a reward graph is saved to
`graphs/` every 25,000 episodes.

Benchmark a checkpoint against fixed opponents and write results to
`benchmarks.csv`:

```bash
python -m training.selfplay.evaluate --episodes 1000 --opponent all --mcts-sims 0
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
`checkpoints/`.  Simply run `python -m training.selfplay.train` again and it will pick up
where it left off.

To resume from a **specific** checkpoint (e.g. after copying one from
another machine), make sure the desired `.pth` file is in the `checkpoints/`
directory and remove any newer checkpoint files so that it becomes the
latest one, then run `python -m training.selfplay.train`.

```bash
# Example: resume from episode 5000
ls checkpoints/          # verify whist_cp_5000.pth exists
python -m training.selfplay.train  # automatically loads the latest checkpoint
```

You can also adjust `TOTAL_EPISODES` in `training/selfplay/train.py` to extend or shorten the
default 1,000,000-episode run.

### Configuration

Key parameters in `training/selfplay/train.py`:

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
python -m training.selfplay.train_multiseat
```

Use a short validation run first:

```powershell
$env:WHIST_TOTAL_EPISODES="1000"
$env:WHIST_CHECKPOINT_EVERY="500"
$env:WHIST_EVAL_EVERY="500"
$env:WHIST_EVAL_EPISODES="100"
python -m training.selfplay.train_multiseat
```

### Strong transformer experiment

`training/selfplay/train_strong.py` is an isolated experiment for stronger strategic play. It
uses the transformer card extractor, lower heuristic shaping, a larger terminal
team-result reward, and promotion by fixed rule-opponent performance. Each
checkpoint receives a fast raw-policy evaluation over balanced team assignments
and a separate hidden-hand MCTS evaluation. The best raw-policy checkpoint is
saved as `checkpoints_strong/best_rule.pth`.

```powershell
python -m training.selfplay.train_strong
```

It writes checkpoints, reward/win-rate graphs, and benchmark scores to its own
`*_strong` paths and never replaces the original or multi-seat runs. By
default it saves a checkpoint every 5,000 episodes, a graph every 10,000
episodes, and logs reward every 250 episodes. Each checkpoint uses a compact
50-game raw benchmark plus a 4-game hidden-hand MCTS spot check.

### Fast competitive experiment

For a faster strategic experiment, use `training/selfplay/train_competitive.py`. It keeps the
MLP policy and four-seat training, but mixes a real fixed rule player into
35% of opponent turns. It evaluates 200 complete balanced games at every
checkpoint and promotes the best rule-player result to
`checkpoints_competitive/best_rule.pth`.

```powershell
python -m training.selfplay.train_competitive
```

Its outputs are isolated under `checkpoints_competitive/`,
`rewards_competitive.csv`, `graphs_competitive/`, and
`competitive_checkpoint_scores.csv`.

### Post-competitive self-play

After competitive training is complete, start the next phase with:

```powershell
python -m training.selfplay.train_post_selfplay
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
python -m training.selfplay.train_anchor_selfplay
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
python -m training.selfplay.train

# Force a specific device
WHIST_DEVICE=cuda python -m training.selfplay.train
WHIST_DEVICE=cpu python -m training.selfplay.train

# Adjust the soft GPU cap if needed
WHIST_GPU_CAP_PERCENT=60 python -m training.selfplay.train
WHIST_GPU_CAP_PERCENT=0 python -m training.selfplay.train
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
