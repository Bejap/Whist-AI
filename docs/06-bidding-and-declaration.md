# Phase 6: Bidding and declaration

## Goal

Learn contract choice, trump, and partner suit while card play is provided by a frozen specialist.

## Why this is separate

A bid is a long-horizon decision. Its quality depends on the resulting contract, declaration, trick play, and settlement. Training bidding at the same time as immature card play makes credit assignment ambiguous: the bidder cannot tell whether a loss came from a bad bid or bad cards.

## Training approach

Start with imitation or search-generated targets where available. Then fine-tune with masked PPO using the Phase 2 game-point reward against a frozen card-play specialist and fixed opponents. Include opening decisions, responses, successful contracts, failed contracts, and pass behavior.

The first reinforcement-learning reward is explicit:

```text
non_terminal_reward = 0.0
```

The terminal reward is the numeric, nolo, or Paskrig result defined in Phase 2.
It rewards higher fulfilled numeric bids and extra tricks, punishes every
missed trick twice, makes failed 13 bids especially costly, and gives nolo
failures a 1.5x penalty. The bid policy learns which legal decisions tend to
produce better game-point outcomes; it does not learn the meaning of the reward
from scratch. Raw settlement is still logged as a secondary evaluation metric.

For an auction decision that is not followed by the learner becoming declarer,
the first implementation still records the final learner settlement, but the
evaluation reports those decisions separately. This prevents us from claiming
that a bid is good merely because a different seat later won. If credit
assignment is too noisy, the next experiment is a counterfactual auction
baseline or imitation pretraining, not an arbitrary additional bid bonus.

The declaration policy should choose trump and partner suit only after a contract makes the choice legal. It must not be merged into the bidding target prematurely.

## Metrics

Track bid frequency by hand strength, contract success by bid, average payment, overbid rate, underbid rate, pass calibration, trump quality, partner-card outcomes, and the difference between declarer and defender performance.

## Exit criteria

- Bidding completes legal auctions without artificial opponent behavior dominating results.
- The specialist is evaluated with fixed card play.
- Contract selection improves payment over a rules baseline on held-out deals.
- Declaration choices are legal and separately measurable.
