# Phase 5: Specialist learning

## Goal

Train card play as a reusable specialist before adding auction decisions.

## Recommended sequence

1. Train against deterministic rules opponents for a correctness baseline.
2. Add random and perturbed-rule opponents for robustness.
3. Add frozen historical policies as they become available.
4. Evaluate on opponents and seeds not used for training.

Use masked PPO initially because the action space is discrete and legal actions vary by position. Keep the policy observation and action-space version in every checkpoint.

## Reward design

Use the Phase 2 terminal game-point reward exactly for the first specialist:

```text
non_terminal_reward = 0.0
```

The terminal reward is the numeric, nolo, or Paskrig game-point result defined
in Phase 2. Numeric contracts reward a fulfilled bid and tricks above seven,
charge two points for every missed trick, and apply the additional locked-in
penalty to failed 13 bids. Nolo contracts use fixed success values and a 1.5x
failure multiplier. Paskrig uses its intentionally asymmetric relative-trick
reward. Raw settlement is retained in every episode record for evaluation.

The model should discover the action strategy from this objective. It should
not be asked to discover the objective itself. Invalid actions are rejected and
fail the legality tests; they are not silently converted into a legal card.

Do not add per-trick rewards in the baseline. A later shaping experiment must
use a documented potential, compare against this no-shaping baseline on
identical seeds, and be discarded if it improves an intermediate metric while
reducing held-out game points or settlement.

## Resource policy

Use bounded rollout and batch sizes. Opponent models stay on CPU unless profiling proves GPU inference is worthwhile. Add a RAM monitor that pauses rollout collection above a configured system-memory threshold and resumes below a lower threshold.

## Exit criteria

- The specialist beats or matches the rules baseline on held-out contracts.
- It performs across declarer and defender roles.
- No observation leakage or invalid action occurs.
- Checkpoint resume is tested.
- Evaluation is stable across at least two independent seed sets.
