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

The model does not invent its own reward. The environment supplies a simple,
explicit terminal game-point reward. The policy discovers which legal actions
produce good outcomes; the project defines what good means below.

All non-terminal decisions receive `0.0`. The terminal reward is calculated
from the final contract and trick result. Raw settlement is still recorded and
remains an evaluation metric, but it is not the training reward in this design.

### Numeric contracts

For a successful numeric contract, the bid contributes one point for each
level above 7, and the trick component counts from 7 through the tricks taken:

```text
bid_value = bid - 7
success_reward = bid_value + (tricks_won - 6)
```

For failed bids from 7 through 12, the bid value is lost and every missed trick
costs two points:

```text
missed = bid - tricks_won
failure_reward = -bid_value - 2 * missed
```

Bid 13 is a locked-in contract. Its failure has an additional six-point
surcharge, so missing by one is worse than its maximum ordinary success value:

```text
failure_reward_for_13 = -6 - 6 - 2 * (13 - tricks_won)
```

| Bid | Tricks | Outcome | Reward |
|---:|---:|---|---:|
| 7 | 7 | Success | +1 |
| 7 | 8 | Success | +2 |
| 8 | 8 | Success | +3 |
| 9 | 9 | Success | +5 |
| 10 | 10 | Success | +7 |
| 12 | 12 | Success | +11 |
| 13 | 13 | Success | +13 |
| 7 | 6 | Missed by 1 | -2 |
| 8 | 7 | Missed by 1 | -3 |
| 8 | 6 | Missed by 2 | -5 |
| 9 | 7 | Missed by 2 | -6 |
| 12 | 10 | Missed by 2 | -9 |
| 13 | 12 | Locked-in contract missed by 1 | -14 |
| 13 | 10 | Locked-in contract missed by 3 | -18 |

### Nolo contracts

A nolo failure ends immediately when its forbidden trick is taken. Therefore,
there is no per-trick failure accumulation. Failure is deliberately 1.5 times
the success value:

| Contract | Success condition | Success reward | Failure condition | Failure reward |
|---|---|---:|---|---:|
| Sol | Declarer takes 0 or 1 trick | +4 | Declarer takes a second trick | -6 |
| Ren sol | Declarer takes 0 tricks | +8 | Declarer takes one trick | -12 |
| Bordlaegger | Declarer takes 0 tricks | +12 | Declarer takes one trick | -18 |

### Paskrig

Paskrig rewards avoiding tricks and intentionally uses an asymmetric reward.
Let `fewest` be the smallest trick count and `most` the largest:

```text
winner_reward = most - fewest
non_winner_reward = -2 * (tricks_won - fewest)
```

Every player tied for the fewest tricks is a winner and receives the same
winner reward. For example, `[1, 1, 2, 9]` produces `[+8, +8, -2, -16]`.
With one winner, `[1, 2, 4, 6]` produces `[+5, -2, -6, -10]`.
These rewards are intentionally not zero-sum: taking many tricks is punished
more strongly than winning with few tricks is rewarded.

An invalid action is a simulator or policy bug: it is rejected and counted in
tests rather than silently replaced. Intermediate reward shaping is not part
of the baseline. The policy is expected to discover bidding, card selection,
risk, and coordination strategies from these terminal outcomes, not to
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
