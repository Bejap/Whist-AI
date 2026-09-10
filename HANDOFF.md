# Whist-AI Handoff

Updated: 2026-09-10

## Resume checkpoint

The portable card-play checkpoint is tracked at:

`checkpoints/esmakker_cardplay_v1/latest.zip`

Current saved state:

- Contract: `7`
- PPO timesteps: `401408`
- PPO updates: `780`
- Checkpoint size: about 2.68 MB

The checkpoint can be resumed with:

```powershell
.venv\Scripts\python.exe train_esmakker_cardplay.py --run-name esmakker_cardplay_v1 --contract 7 --resume --timesteps 500000 --device cuda --gpu-memory-fraction 0.30 --n-steps 2048 --batch-size 512 --checkpoint-every 50000
```

Use a new chat in the repository and ask the agent to read this file first.

## Important status

- The team-trick accounting bug is fixed in `esmakker_cardplay_env.py`.
- The regression test is in `test_esmakker_cardplay_env.py`.
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
.venv\Scripts\python.exe -m unittest -v test_esmakker_cardplay_env.py
```
