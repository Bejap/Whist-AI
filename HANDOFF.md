# Whist-AI Handoff

Updated: 2026-09-10

## Resume checkpoint

The portable card-play checkpoint is tracked at:

`checkpoints/esmakker_cardplay_v1/latest.zip`

Training code: `training/cardplay/train_esmakker_cardplay.py`

Current saved state:

- Contract: `7`
- PPO timesteps: `401408`
- PPO updates: `780`
- Checkpoint size: about 2.68 MB

The checkpoint can be resumed with:

```powershell
.venv\Scripts\python.exe -m training.cardplay.train_esmakker_cardplay --run-name esmakker_cardplay_v1 --contract 7 --resume --timesteps 500000 --device cuda --gpu-memory-fraction 0.30 --n-steps 2048 --batch-size 512 --checkpoint-every 50000
```

Use a new chat in the repository and ask the agent to read this file first.

## Important status

- The team-trick accounting bug is fixed in `training/cardplay/esmakker_cardplay_env.py`.
- The regression test is in `training/cardplay/test_esmakker_cardplay_env.py`.
- The fix uses unique team seats, so the learner is not counted twice when also holding the partner card.
- The focused card-play tests passed after the fix.
- Commit containing the fix: `8623b1c`.
- The currently running training processes were started before the fix and must be stopped and restarted before training with corrected rewards.
- The latest checkpoint was produced by the pre-fix process, so its weights were trained partly with the old reward calculation. Evaluation with the fixed environment is valid, but future training should resume only after restarting the process.

## Latest benchmark

The latest saved checkpoint was evaluated over 500 fresh seeded games using deterministic actions and the corrected metric:

- Success rate: `56.6%`
- Average true team tricks: `7.106`
- Average payment: `-0.380`
- Maximum team tricks: `13`

These numbers are a baseline, not proof of improvement. The earlier 350,208-timestep checkpoint measured 58.4% success, 7.194 average team tricks, and -0.242 average payment on the same seed range.

## Metrics caution

There are currently two `train_esmakker_cardplay.py` processes using the same run directory. That can race while rewriting `graphs/esmakker_cardplay_v1/cardplay_metrics.csv`; the CSV should not be treated as authoritative until only one trainer process is running. The checkpoint file is the authoritative resumable model state.

## Validation

Run the focused tests with:

```powershell
.venv\Scripts\python.exe -m unittest -v training.cardplay.test_esmakker_cardplay_env
```

## Intended full-game architecture

The long-term goal is a phase-separated full-game agent, not one monolithic
policy with bidding, declaration, and card-play actions in a single 71-action
head. A single policy has too many unrelated decisions and parameters to track
at once, which makes training and diagnosis harder.

The intended architecture is a composite controller that delegates decisions
by game phase:

1. A bidding specialist chooses bids or pass.
2. A declaration specialist chooses trump and the partner suit.
3. A card-play specialist chooses legal cards during trick play.
4. The full-game controller owns the rules flow, action masks, phase routing,
	and final settlement.

Training order:

1. Keep the fixed-contract card-play specialist as the first component. The
	current checkpoint is the card-play baseline, though it currently supports
	contract 7 only.
2. Train bidding and declaration against a frozen card-play specialist (or a
	deterministic rules player where the specialist is unavailable), so bid
	quality is measured by the resulting contract settlement.
3. Integrate the specialists behind one full-game controller and benchmark the
	combined agent against fixed-rule opponents.
4. Only after the modular system is stable, consider joint fine-tuning.

The existing `training/esmakker/train_esmakker.py` and `EsmakkerEnv` implement
the monolithic full-game experiment. They are useful reference code and rules
coverage, but they are not the target architecture and should not be resumed
as the main step-2 training path. The fixed-contract card-play checkpoint
cannot be loaded directly into that trainer because its policy head has a
different action space.

## Card-play specialist upgrade

The card-play environment now supports `--contract all`, which samples
numeric contracts 7 through 13 during training. Its observation includes the
existing hand/trick state plus a 52-bit played-card history, a 208-bit
player-card ownership history, and 16 player-by-suit void indicators. The
card-play MLP was enlarged to `[512, 512, 256]` so it has more capacity for
tracking those features.

This changes the observation and policy architecture. Do not resume the old
`esmakker_cardplay_v1/latest.zip` checkpoint with the upgraded trainer. Keep it
as the contract-7 baseline and start a new run name:

```powershell
.venv\Scripts\python.exe -m training.cardplay.train_esmakker_cardplay `
	--run-name esmakker_cardplay_v2 `
	--contract all `
	--timesteps 1000000 `
	--device cuda `
	--gpu-memory-fraction 0.30 `
	--n-steps 2048 `
	--batch-size 512 `
	--checkpoint-every 50000
```

The v2 specialist should be evaluated across contracts before it is used as
the frozen card-play component for separate bidding/declaration training.

The card-play curriculum now randomizes the declarer independently from the
learning seat. The learner therefore trains as both declarer and defender;
terminal payment is always taken from the learning seat's settlement. Resume
the compatible v2 checkpoint after this change so it adapts to both roles.

When `--contract all` is used, contracts are sampled with Zipf weighting in
the order `9, 8, 7, 10, 11, 12, 13`: weights are `1, 1/2, 1/3, ... 1/7`.
This intentionally gives the central, commonly playable contracts more
training exposure while retaining all numeric contracts. A running process
must be restarted before it uses this changed distribution.

The deterministic benchmark command is:

```powershell
.venv\Scripts\python.exe -m training.cardplay.evaluate_cardplay `
	--checkpoint checkpoints/esmakker_cardplay_v2/latest.zip `
	--contract all `
	--episodes 500 `
	--device cuda
```

It writes per-game results to
`graphs/esmakker_cardplay_v2/cardplay_benchmark_games.csv` and aggregate
results by contract to
`graphs/esmakker_cardplay_v2/cardplay_benchmark_summary.csv`. Keep the seed
and episode count fixed when comparing checkpoints.
