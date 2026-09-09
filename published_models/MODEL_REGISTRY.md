# Published Models

These checkpoints are the portable models intended for use on another machine.
Large training checkpoint directories remain local-only.

| File | Training line | Notes |
|---|---|---|
| `original_mlp_ep360000.pth` | Original single-seat MLP | Original baseline run |
| `multiseat_mlp_ep250000.pth` | Multi-seat MLP | Shared policy across four seats |
| `competitive_mlp_best.pth` | Competitive MLP | Best recorded result against the rule player |
| `strong_transformer_best.pth` | Strong transformer | Best recorded strong-experiment checkpoint |

The checkpoints are stored with Git LFS. After cloning, install Git LFS and run:

```powershell
git lfs install
git lfs pull
```

The current gameplay loader still looks in `checkpoints/`. Copy the desired
published checkpoint there, or update the loader to select a published model.
