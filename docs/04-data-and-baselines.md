# Phase 4: Data and baselines

## Goal

Establish opponents and measurements before learning begins.

## Baselines

Implement and freeze:

- A legal random player.
- A deterministic rules player.
- A stronger heuristic player with hand strength, contract risk, and trick logic.
- A search or rollout player for small endgame positions where practical.
- Historical checkpoint opponents once compatible checkpoints exist.

## Data

Generate seeded self-play and baseline games. Store the seed, rules revision, policy revision, seat, phase decisions, contract, tricks, payment, and completion status. For bidding, retain the hand features available at the decision and the final settlement.

Human games may later provide imitation data, but they are supplementary. They should not replace a strong legal baseline or settlement-based evaluation.

## Evaluation design

Use separate training, validation, and held-out test seeds. Report completion rate, average payment, contract success, trick difference, bid calibration, overbid rate, and seat balance. Always compare the same number of games and same seeds between checkpoints.

## Exit criteria

- Baseline policies complete games legally.
- Baseline strength is measured on held-out seeds.
- Data schemas and replay files are versioned.
- A benchmark command produces machine-readable and human-readable reports.
