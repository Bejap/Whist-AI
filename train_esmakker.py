"""Train an Esmakker Whist policy against rule-based opponents."""

import argparse
from pathlib import Path

from sb3_contrib import MaskablePPO

from esmakker_rl_env import EsmakkerEnv


CHECKPOINT_DIR = Path("checkpoints") / "esmakker"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=250_000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = CHECKPOINT_DIR / "latest.zip"
    env = EsmakkerEnv()

    if args.resume and checkpoint.exists():
        model = MaskablePPO.load(checkpoint, env=env, device=args.device)
        print(f"Resuming {checkpoint}")
    else:
        model = MaskablePPO(
            "MlpPolicy",
            env,
            n_steps=512,
            batch_size=128,
            n_epochs=4,
            gamma=0.995,
            learning_rate=3e-4,
            ent_coef=0.01,
            policy_kwargs={"net_arch": [256, 256]},
            device=args.device,
            verbose=1,
        )

    model.learn(total_timesteps=args.timesteps, reset_num_timesteps=not args.resume)
    model.save(CHECKPOINT_DIR / "latest")
    print(f"Saved {checkpoint}")


if __name__ == "__main__":
    main()