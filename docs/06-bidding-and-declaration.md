# Phase 6: Bidding and declaration

## Goal

Learn contract choice, trump, and partner suit while card play is provided by a frozen specialist.

## Why this is separate

A bid is a long-horizon decision. Its quality depends on the resulting contract, declaration, trick play, and settlement. Training bidding at the same time as immature card play makes credit assignment ambiguous: the bidder cannot tell whether a loss came from a bad bid or bad cards.

## Training approach

Start with imitation or search-generated targets where available. Then fine-tune with masked PPO using settlement-based rewards against a frozen card-play specialist and fixed opponents. Include opening decisions, responses, successful contracts, failed contracts, and pass behavior.

The declaration policy should choose trump and partner suit only after a contract makes the choice legal. It must not be merged into the bidding target prematurely.

## Metrics

Track bid frequency by hand strength, contract success by bid, average payment, overbid rate, underbid rate, pass calibration, trump quality, partner-card outcomes, and the difference between declarer and defender performance.

## Exit criteria

- Bidding completes legal auctions without artificial opponent behavior dominating results.
- The specialist is evaluated with fixed card play.
- Contract selection improves payment over a rules baseline on held-out deals.
- Declaration choices are legal and separately measurable.
