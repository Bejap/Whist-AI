# Phase 2: Rules and game specification

## Goal

Turn the written Esmakker rules into a testable contract between the simulator, policies, and evaluators.

## Required specification

Document and test:

- Deal, dealer rotation, turn order, and seed behavior.
- Bid ordering and pass semantics.
- Numeric contracts from 7 through 13.
- Sol, Ren sol, and Bordlaegger nolo contracts.
- Paskrig after four passes.
- Trump selection and legal partner-suit selection.
- Secret partner card and its reveal rule.
- Ace ranking exceptions.
- Trick legality and winner selection.
- Settlement, zero-sum payments, and terminal outcomes.

## Design decision

The rules engine, not the learning environment, owns all game transitions. The environment adapts the engine to a reinforcement-learning API and must not duplicate rule logic.

## Reward contract

The model does not invent its own reward. The environment supplies a reward that
encodes the project objective: maximize the learner's long-run settlement while
playing legal actions.

The rules engine first computes the complete settlement. The canonical terminal
signal is the learner seat's actual payment, including whether the learner was
the declarer, the declarer's partner, a defender, or a Paskrig winner:

```text
terminal_reward = learner_payment / 25.0
```

The scale changes magnitude only; it does not change which outcome is better.
The unscaled payment remains the authoritative evaluation metric. No separate
hand-coded reward is allowed to override settlement. A successful contract is
therefore not automatically good if its payment is worse than another legal
outcome, and taking a trick is not automatically good in a nolo contract.

All non-terminal decisions receive `0.0` in the first implementation. An
invalid action is a simulator or policy bug: it is rejected and counted in
tests rather than silently replaced. If a bounded invalid-action penalty is
needed for an experiment, it must be documented and evaluated separately from
the canonical reward.

Intermediate trick shaping is not part of the canonical objective. It may be
introduced later only as a potential-based signal, with an ablation proving
that it does not make the policy optimize tricks instead of settlement. The
policy is expected to discover bidding, card-selection, risk, and coordination
strategies from observations and terminal outcomes; it is not expected to
discover what the reward should mean.

## Test strategy

Use deterministic unit tests for every settlement branch and edge case. Add property tests for:

- Every completed game has 13 tricks.
- Every played card is unique.
- Every action accepted by the environment is legal.
- Payments sum to zero.
- The same seed and action sequence produce the same result.

## Exit criteria

- The rules document matches the executable engine.
- Edge cases have focused regression tests.
- A deterministic random-policy game suite completes without invalid state.
