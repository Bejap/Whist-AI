# Phase 3: Simulator and interfaces

## Goal

Create a stable simulator boundary that every policy and evaluator can use.

## Architecture

The simulator exposes a full-information internal state only to the rules engine. Each seat receives a perspective observation containing only information legal for that seat to know. The environment provides an action mask and translates one policy decision at a time.

The controller interface should be phase explicit:

```text
choose_bid(observation, legal_bids) -> bid_or_pass
choose_declaration(observation, legal_trump_suits, legal_partner_suits) -> choice
choose_card(observation, legal_cards) -> card
```

A policy may be neural, rule-based, random, or search-based as long as it obeys the same interface.

## Observation requirements

Observations should identify phase, acting seat, hand, visible played cards, current contract, trump, known partner information, trick state, and public history. They must not reveal unplayed opponent cards or hidden partner identity.

## Action masking

Masks are mandatory. Invalid actions should be rejected in tests and never silently converted into another action during training, because silent replacement hides policy bugs.

## Performance and resources

Start with one environment and CPU opponent inference for correctness. Add parallel environments only after measuring throughput and RAM. Resource controls must cover both GPU VRAM and system RAM; a VRAM fraction alone is not a RAM safeguard.

## Exit criteria

- The phase interface works with random, rules, and stub neural policies.
- Observation leakage tests pass.
- Action legality tests pass.
- A full deterministic game can be replayed from a seed and action log.
