# Potential AI Improvements

Ideas for making the Whist AI smarter, roughly ordered from easiest to hardest.

---

## 1. Probabilistic hand inference (opponent modelling)
Maintain a probability distribution over unobserved opponent hands, updated
via Bayes' rule as cards are played.  The agent can condition its policy on
these beliefs to, e.g., avoid leading into a known void.

## 2. Phase-separated full-game agent
The full-game agent should be composed from specialist policies rather than a
single monolithic policy that must learn bidding, declaration, and card play
through one large action head.

The controller should delegate by phase:

- **Bidding specialist** – chooses a contract or pass.
- **Declaration specialist** – chooses trump and the partner suit.
- **Card-play specialist** – chooses legal cards during trick play.
- **Full-game controller** – owns rules flow, masks, phase routing, and final
  settlement.

Training should proceed in stages. First preserve the fixed-contract
card-play policy as a frozen baseline. Then train bidding and declaration
against that specialist so a bid is judged by the resulting contract
settlement. Finally integrate the specialists and benchmark the combined agent
against fixed-rule opponents before attempting joint fine-tuning.

The existing `training/esmakker/train_esmakker.py` is a monolithic reference
implementation with a 71-action policy. It should not be treated as the target
step-2 trainer or resumed for this architecture.

## 3. Improved reward shaping
Current shaping rewards are hand-coded heuristics.  Alternatives:

- **Potential-based shaping** – derive bonuses from a value-function estimate
  so they are guaranteed not to change the optimal policy.
- **Counterfactual multi-agent credit assignment (COMA)** – reward each
  player based on its marginal contribution to the team outcome.

## 4. Team coordination signal
Because two players share a team, introduce an explicit *communication
channel* (e.g. a dedicated output head) so teammates can signal information
(e.g. strength in a suit) without breaking the rules of the game.

## 5. ELO-based curriculum
Track an ELO score for each checkpoint in the league.  Always select
opponents within a small ELO window of the current agent so training
difficulty self-adjusts as the agent improves.

## 6. Endgame tablebase / exact solver
For late-game positions with few cards left, use a depth-limited perfect-play
solver to provide exact targets.  These can be used both during evaluation and
as additional supervised training data for the value head.

## 7. Counterfactual data augmentation
Augment trajectories by swapping equivalent cards/suits in symmetric states and
relabeling actions accordingly.  This increases sample efficiency and can make
the policy less sensitive to superficial card-identity variance.

## 8. Distillation for fast deployment
After training a large recurrent + search-assisted policy, distill it into a
smaller feed-forward student model for faster CPU inference while retaining
most of the playing strength.

## 9. Risk-sensitive objective
Optimise not only expected trick count, but also variance-aware utility (e.g.
CVaR).  This can produce safer play under uncertainty, especially in close
matches where avoiding catastrophic tricks matters more than average score.

## 10. Opponent-style conditioning
Train a shared policy with a latent opponent-style embedding inferred online.
Conditioning on style (aggressive trumping, conservative leads, etc.) can
improve adaptation during a match without retraining.
