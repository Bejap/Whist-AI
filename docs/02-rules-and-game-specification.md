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
