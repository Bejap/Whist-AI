# Training plan: specialists before integration

This is the authoritative execution plan for model training. A training run is
not considered progress toward the final agent unless its phase gate below is
met and its artifacts are recorded.

## Current status

- [x] Rules engine and terminal settlement tests
- [x] Public-information observation with card ownership and void suits
- [x] Resumable checkpoints with optimizer state
- [x] Fixed rule opponents and held-out evaluation scaffold
- [x] Historical 138-feature checkpoint retained
- [ ] Card-play specialist
- [ ] Counterfactual contract evaluator
- [ ] Legacy checkpoint opponent adapter
- [ ] Bidding and declaration specialist
- [ ] Integrated controller benchmark

The existing `observation_v3` joint run is a historical full-game baseline. It
must not be used as evidence that the specialist plan has succeeded.

## Non-negotiable reward policy

Use the authoritative terminal Esmakker settlement from the rules specification.
Do not add a reward bonus for bidding 8 or 9, do not penalize bidding 7 merely
because it is 7, and do not add per-card or per-trick reward in the baseline.
Any shaping experiment requires an explicit comparison against the terminal
reward on identical seeds and must be rejected if held-out settlement worsens.

## Phase A: card-play specialist

Train card play with the auction and declaration fixed by the scenario. The
policy receives the legal public observation and selects only legal cards.
Bidding is not learned in this phase.

Required scenario coverage:

- Numeric contracts 7 through 13
- Sol, Ren sol, Bordlaegger, and Paskrig
- Declarer, partner, and defender roles
- Rotating dealer and fixed held-out seeds
- Conservative, balanced, aggressive, and random legal opponents

Required logged fields per episode:

- seed, rules version, policy version, scenario id
- contract, declarer, partner, seat role
- hand strength and estimated tricks for analysis only
- tricks per seat, team tricks, settlement, completion
- illegal action count and decision count

Exit gate:

- No illegal actions or incomplete games
- Checkpoint resume passes
- Held-out settlement matches or beats the fixed rules card-play baseline
- Results are reported by contract and seat role on two independent seed sets

## Phase B: counterfactual contract evaluation

For the same deal and opponent policies, evaluate legal alternatives such as
pass, 7, 8, and 9. Keep the deal, fixed card-play policy, declaration policy,
and opponent seeds identical. Store one row per candidate action with the final
settlement and trick outcome.

This evaluator is an analysis tool first. It does not change the official
training reward.

Required outputs:

- candidate action and expected settlement
- actual tricks and contract success
- overbid and underbid classification
- hand strength and estimated tricks
- comparison with the policy's chosen action

## Phase C: legacy and snapshot opponents

Implement a compatibility adapter for the old `balanced_policy.pt` using its
138-feature encoder. The adapter is frozen and may only be used as an opponent.
It must never be loaded into the 584-feature policy. Add newer checkpoints and
self-play snapshots only after the legacy match is reproducible.

## Phase D: bidding and declaration specialist

Freeze the Phase A card-play specialist in evaluation mode. Train bidding and
declaration separately against fixed rule opponents and the frozen card-play
specialist. Log bid calibration by hand strength, contract success, expected
tricks, overbid rate, underbid rate, and role.

Exit gate:

- Legal auctions and declarations
- Card-play specialist remains unchanged during bidding updates
- Counterfactual evaluator shows improved action selection on held-out deals
- Payment improves over the rule bidding baseline without a reward hack

## Phase E: integrated agent and population

Compose the specialists through the phase-aware controller. Benchmark against:

- Conservative rule policy
- Balanced rule policy
- Aggressive rule policy
- Random legal policy
- Frozen legacy `balanced_policy.pt`
- Frozen newer checkpoints
- Self-play snapshots, when available

Use identical held-out seeds, seat rotation, confidence intervals where
practical, and a machine-readable report. Joint fine-tuning is optional and
comes only after the integrated baseline passes.

## Training commands and ownership

The user launches substantive training. Every command must name:

- A new checkpoint path for the active phase
- Metrics and decision paths
- Seed and held-out seed set
- CUDA device and safety limits
- Resume behavior

Never resume a checkpoint from another observation or action-space version.
Never start Phase D while the Phase A exit gate is unchecked.
