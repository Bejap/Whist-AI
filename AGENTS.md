# Agent instructions for Whist-AI

## Follow the plan

The authoritative project plan is in `docs/README.md` and the phase documents under `docs/`.

- Follow the phases in order unless a documented dependency requires a change.
- Read the relevant phase document before implementing that phase.
- Treat each phase exit criterion as required work, not optional guidance.
- Keep the rules engine authoritative and preserve the phase-separated architecture.
- Do not start expensive training before the required simulator, baseline, and evaluation checks pass.
- Keep changes focused on the current phase and update the relevant documentation when a decision changes.

## Continue through completion

Carry work through implementation, tests, validation, and documentation. Do not stop at a plan or partial implementation when the next step is feasible. If blocked, identify the concrete blocker, make all non-blocked progress possible, and report the exact action needed to continue.

Before finishing a task:

1. Run the narrowest relevant tests or checks.
2. Inspect the resulting diff and repository status.
3. Update phase documentation or the handoff notes when needed.
4. State what passed, what remains, and any operational constraints.

## Git workflow

Commit and push meaningful progress throughout the process.

- Work on the designated feature branch; do not commit directly to the default branch.
- Make small, coherent commits at phase boundaries or after a complete tested slice.
- Use descriptive commit messages that identify the phase and outcome.
- Push each meaningful commit after local validation so progress is backed up remotely.
- Never use destructive history rewrites, force-pushes, or commands that discard user changes without explicit approval.
- Do not commit secrets, credentials, local virtual environments, generated checkpoints, or unreviewed large artifacts.
- Before committing, review `git diff --check`, `git status`, and the staged diff.

## Resource discipline

Training and evaluation must respect system RAM and GPU limits. Prefer bounded workloads, CPU opponent inference, reproducible seeds, and explicit pause or stop behavior for resource-heavy jobs. Do not launch a second expensive process while another one is active without checking resource headroom.

## Metrics and graphs

- Every substantive training or evaluation run must write a machine-readable metrics file and maintain a corresponding graph in `graphs/`.
- Metrics must be flushed during the run so the current progress can be inspected while training is active.
- This applies to every training style: declarer, partner, defender, mixed-role, curriculum, self-play, population, and evaluation.
- Graphs must show the primary phase objective, not only aggregate episode success. For declarer training, always include declarer-team settlement and win rate by contract; for other styles, graph that style's own role/team objective separately.
- Use a distinct checkpoint, metrics path, and graph path for each experiment. Never mix rows from different objectives, roles, or checkpoints in one graph.
- When reporting progress, include the latest episode, metrics path, graph path, primary objective, and the command used to refresh the graph.
- Refresh the graph after meaningful progress and before making a training decision or advancing a phase gate.

## Training ownership

- The user launches and stops substantive training runs from their own terminal.
- The agent must not start long-running or expensive training autonomously.
- The agent may prepare commands, inspect training output, generate graphs, and run short smoke tests when explicitly requested.
- Before suggesting a training command, include the checkpoint path, metrics path, CUDA device, RAM limits, GPU temperature limits, and VRAM limit.
