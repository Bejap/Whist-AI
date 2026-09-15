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

The terminal reward should be derived from the learner's actual settlement or team result. Small intermediate trick signals may improve learning, but they must be audited against the terminal objective and never reward illegal or information-leaking behavior.

## Resource policy

Use bounded rollout and batch sizes. Opponent models stay on CPU unless profiling proves GPU inference is worthwhile. Add a RAM monitor that pauses rollout collection above a configured system-memory threshold and resumes below a lower threshold.

## Exit criteria

- The specialist beats or matches the rules baseline on held-out contracts.
- It performs across declarer and defender roles.
- No observation leakage or invalid action occurs.
- Checkpoint resume is tested.
- Evaluation is stable across at least two independent seed sets.
