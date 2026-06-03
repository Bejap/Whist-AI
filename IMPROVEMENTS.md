# Potential AI Improvements

Ideas for making the Whist AI smarter, roughly ordered from easiest to hardest.

---

## 1. Smarter opponent policy during self-play
Currently opponents use the same model weights or play randomly.  Training
against a mix of skill levels (e.g. increasing epsilon decay over training)
produces a more robust agent.  A *league* approach—keeping a pool of past
checkpoints as opponents—prevents the policy from overfitting to one playstyle.

## 2. Richer observations
The current 171-bit vector omits information that a skilled human uses:

- **Per-player played-card tracking** – instead of a single `played_cards`
  bitmap, keep one 52-bit plane per seat so the model knows *who* discarded
  what.
- **Lead-suit flag** – one-hot of the lead suit for the current trick.
- **Trump exhaustion** – a per-seat flag indicating whether that player has
  run out of trump (inferable from play history).
- **Trick-position index** – which position in the trick the current player
  occupies (0 = lead, 1, 2, 3), since strategy differs strongly by position.

## 3. Recurrent architecture (LSTM / GRU)
Replace the MLP policy with an LSTM.  A recurrent model can build an implicit
memory of all cards played so far without needing the full 52-bit history
planes, and naturally handles partial observability.

## 4. Transformer-based card encoder
Represent each card as an embedding rather than a one-hot bit.  A small
Transformer over the 13-card hand (and the trick so far) learns richer
relational reasoning (e.g. "I hold the second-highest trump").

## 5. Monte Carlo Tree Search (MCTS) at inference time
At test time, run MCTS using the trained value + policy heads as priors.
This costs more computation but can significantly improve play quality,
especially near the end of the game when the tree is shallow.

## 6. Probabilistic hand inference (opponent modelling)
Maintain a probability distribution over unobserved opponent hands, updated
via Bayes' rule as cards are played.  The agent can condition its policy on
these beliefs to, e.g., avoid leading into a known void.

## 7. Bidding / contract phase
Real Whist variants include a bidding phase where players declare how many
tricks their team will win.  Training an agent that optimises for making
(or defeating) a contract rewards more precise, goal-directed play.

## 8. Improved reward shaping
Current shaping rewards are hand-coded heuristics.  Alternatives:

- **Potential-based shaping** – derive bonuses from a value-function estimate
  so they are guaranteed not to change the optimal policy.
- **Counterfactual multi-agent credit assignment (COMA)** – reward each
  player based on its marginal contribution to the team outcome.

## 9. Team coordination signal
Because two players share a team, introduce an explicit *communication
channel* (e.g. a dedicated output head) so teammates can signal information
(e.g. strength in a suit) without breaking the rules of the game.

## 10. ELO-based curriculum
Track an ELO score for each checkpoint in the league.  Always select
opponents within a small ELO window of the current agent so training
difficulty self-adjusts as the agent improves.
