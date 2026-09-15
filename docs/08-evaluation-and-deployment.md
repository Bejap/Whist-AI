# Phase 8: Evaluation and deployment

## Goal

Make progress measurable and produce a usable agent without conflating training output with playing strength.

## Evaluation matrix

Run the integrated agent against:

- Fixed rules opponents.
- Random opponents as a sanity check.
- Historical checkpoints.
- Search or stronger heuristic opponents when available.

Use fixed held-out seeds, enough games for uncertainty estimates, and repeated runs when differences are small. Report confidence intervals where practical.

## Deployment

The first deployment target is a local playable table. It should expose the same controller interface used in evaluation, record replays, and show which specialist made each decision. Human-facing explanations are generated from logs and state, not injected into the policy observation.

## Operational safeguards

- Run evaluation on CPU by default when system RAM is constrained.
- Use a RAM threshold with pause/resume for long training jobs.
- Keep checkpoints immutable after publication.
- Keep a manifest linking each result to code revision and configuration.

## Exit criteria

- A reproducible benchmark report exists for the integrated agent.
- Replays can be inspected and rerun.
- The local playable program uses the same tested controller.
- Published checkpoints and model metadata are documented.
