# Published Models

These checkpoints are the portable models intended for use on another machine.
Large training checkpoint directories remain local-only.

| File | Training line | Notes |
|---|---|---|
| `original_mlp_ep360000.pth` | Original single-seat MLP | Original baseline run |
| `multiseat_mlp_ep250000.pth` | Multi-seat MLP | Shared policy across four seats |
| `competitive_mlp_best.pth` | Competitive MLP | Best recorded result against the rule player |
| `strong_transformer_best.pth` | Strong transformer | Best recorded strong-experiment checkpoint |
| `esmakker_cardplay_v2_best.zip` | Esmakker card-play specialist v2 | Best measured v2 checkpoint against the archived learned-opponent table; 2,101,248 PPO timesteps |

The checkpoints are stored with Git LFS. After cloning, install Git LFS and run:

```powershell
git lfs install
git lfs pull
```

The current gameplay loader still looks in `checkpoints/`. Copy the desired
published checkpoint there, or update the loader to select a published model.

The Esmakker v2 archive uses the 422-feature observation space and 52-card
action space. Benchmark context for the selected checkpoint used 100 games per
contract and the archived learned checkpoint at 1,701,888 timesteps as the
learned-opponent table. It achieved -0.801 average payment and 42.857% learner
success against that table, outperforming the 2,201,600-timestep checkpoint
in that comparison.
