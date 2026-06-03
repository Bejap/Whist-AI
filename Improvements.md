# Future Improvements

## 1. GPU Training
Before enabling GPU training, migrate the following components:

### RecurrentPPO (LSTM)
- Replace PPO with RecurrentPPO from sb3-contrib
- Requires: `pip install sb3-contrib`
- Benefit: agent can remember context across timesteps within a trick
- Note: significantly slower on CPU — only enable with a supported GPU backend (CUDA/ROCm)

### Transformer Feature Extractor
- Custom `BaseFeaturesExtractor` subclass using `torch.nn.MultiheadAttention`
- Processes the 52-card hand as a sequence of card embeddings
- Benefit: better at learning card relationships and suit patterns
- Note: attention mechanism is very slow on CPU — GPU only

### MCTS Inference (play.py)
- Monte Carlo Tree Search with N simulations per move
- Use the trained policy as a rollout heuristic
- Benefit: significantly stronger play at inference time, no retraining needed
- Note: 64 sims per move × 52 moves = ~3300 extra simulations per episode if used during training
