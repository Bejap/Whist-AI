# Whist AI redesign

This directory defines the new build in phases. The old training branch remains available as a historical baseline; this branch starts with design and verification before expensive training.

## Phase order

1. [Vision and scope](01-vision-and-scope.md)
2. [Rules and game specification](02-rules-and-game-specification.md)
3. [Simulator and interfaces](03-simulator-and-interfaces.md)
4. [Data and baselines](04-data-and-baselines.md)
5. [Specialist learning](05-specialist-learning.md)
6. [Bidding and declaration](06-bidding-and-declaration.md)
7. [Full-game integration](07-full-game-integration.md)
8. [Evaluation and deployment](08-evaluation-and-deployment.md)

A phase is complete only when its exit criteria and tests pass. Training is not considered progress by itself; a checkpoint must beat a fixed baseline on a held-out evaluation with reproducible seeds.

## Core architecture

The target is a phase-separated controller:

- The rules engine is deterministic and owns legality, hidden information, trick flow, and settlement.
- A bidding specialist selects a contract or pass.
- A declaration specialist selects trump and partner suit.
- A card-play specialist selects legal cards.
- The controller routes observations and action masks by phase.
- Evaluation compares the complete system against fixed rule and historical opponents.

The first implementation should use one shared card-play policy across roles, then compare that against role-specific policies only after the shared baseline is trustworthy.

## Working agreement

- Keep old checkpoints and generated results as historical artifacts until the new system has a replacement benchmark.
- Make every environment change with a focused regression test.
- Keep training and evaluation commands separate.
- Use CPU opponent inference by default to limit system RAM and GPU contention.
- Record configuration, code revision, seed, opponent set, and checkpoint path for every evaluation.
