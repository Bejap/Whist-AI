# Phase 1: Vision and scope

## Goal

Build a strong, inspectable Esmakker Whist AI that learns the actual game rather than optimizing an accidental simulator artifact.

The first target is a complete playable agent with reliable card play and understandable bidding. Strength comes after correctness and measurement.

## Principles

- Rules are authoritative and deterministic.
- Hidden information must remain hidden from the acting policy.
- Every action is masked to legal choices.
- Rewards reflect settlement and team outcome, not arbitrary action preferences.
- Specialists are trained separately before joint fine-tuning.
- A baseline and held-out evaluation are required before claiming improvement.

## Scope

In scope: four-player Esmakker rules, bidding, declaration, partner-card mechanics, trick play, settlement, self-play, rule opponents, historical opponents, reproducible evaluation, and a playable interface.

Out of scope for the first version: language explanations inside the policy, online learning during a human match, large-scale distributed training, and a monolithic policy that learns every phase from scratch.

## Success definition

The project is successful when the integrated agent completes legal games, beats the fixed rule baseline on held-out seeds, remains competitive against historical checkpoints, and exposes phase-specific metrics that explain wins and losses.

## Exit criteria

- The scope and success metrics are accepted.
- The rules variant is frozen in the game specification.
- No training run starts before Phase 3 tests and Phase 4 baselines exist.
