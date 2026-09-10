"""Train fixed-contract Esmakker card play before introducing bidding."""

import argparse
import csv
from pathlib import Path

import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback

from esmakker_cardplay_env import EsmakkerCardPlayEnv


class CardPlayCheckpointCallback(BaseCallback):
    """Save a resumable card-play checkpoint at regular rollout boundaries."""

    def __init__(self, checkpoint_dir, every=50_000, verbose=0):
        super().__init__(verbose)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.every = every
        self.next_checkpoint = every

    def _on_training_start(self):
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.next_checkpoint = ((self.model.num_timesteps // self.every) + 1) * self.every

    def _on_step(self):
        return True

    def _on_rollout_end(self):
        if self.model.num_timesteps >= self.next_checkpoint:
            self.model.save(self.checkpoint_dir / "latest")
            print(f"Card-play checkpoint saved: {self.checkpoint_dir / 'latest.zip'}")
            self.next_checkpoint = ((self.model.num_timesteps // self.every) + 1) * self.every


class CardPlayMetricsCallback(BaseCallback):
    def __init__(self, metrics_path, verbose=0):
        super().__init__(verbose)
        self.metrics_path = Path(metrics_path)
        self.rows = []

    def _on_training_start(self):
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        if self.metrics_path.exists():
            with self.metrics_path.open(newline="", encoding="utf-8") as handle:
                self.rows = list(csv.DictReader(handle))

    def _on_step(self):
        completed_round = False
        for info, done in zip(self.locals["infos"], self.locals["dones"]):
            if not done:
                continue
            completed_round = True
            settlement = info.get("settlement")
            payment = settlement.payments[info["learning_player"]] if settlement else 0.0
            self.rows.append({
                "episode": len(self.rows) + 1,
                "timesteps": self.num_timesteps,
                "contract": info["contract"],
                "success": int(settlement.contract_succeeded) if settlement else 0,
                "payment": payment,
                "team_tricks": info["team_tricks"],
            })
        if completed_round or self.num_timesteps % 1000 == 0:
            self._write()
        return True

    def _write(self):
        fields = ["episode", "timesteps", "contract", "success", "payment", "team_tricks"]
        with self.metrics_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self.rows)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", choices=[str(number) for number in range(7, 14)], default="7")
    parser.add_argument("--run-name", default="esmakker_cardplay_v1")
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=50_000)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.30)
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.run_name or Path(args.run_name).name != args.run_name:
        raise ValueError("--run-name must be a simple directory name")
    if args.n_steps <= 0 or args.batch_size <= 0 or args.n_steps % args.batch_size != 0:
        raise ValueError("--n-steps must be positive and divisible by --batch-size")
    if args.checkpoint_every <= 0:
        raise ValueError("--checkpoint-every must be positive")
    if not 0.0 < args.gpu_memory_fraction <= 1.0:
        raise ValueError("--gpu-memory-fraction must be greater than 0 and at most 1")
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)
        print(f"CUDA VRAM cap: {args.gpu_memory_fraction:.0%} ({torch.cuda.get_device_name(0)})")

    checkpoint_dir = Path("checkpoints") / args.run_name
    graph_dir = Path("graphs") / args.run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_dir / "latest.zip"
    env = EsmakkerCardPlayEnv(contract=args.contract)

    if args.resume and checkpoint.exists():
        model = MaskablePPO.load(checkpoint, env=env, device=args.device)
        print(f"Resuming {checkpoint}")
    else:
        model = MaskablePPO(
            "MlpPolicy",
            env,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=4,
            gamma=0.995,
            learning_rate=3e-4,
            ent_coef=0.01,
            policy_kwargs={"net_arch": [256, 256]},
            device=args.device,
            verbose=1,
        )

    model.learn(
        total_timesteps=args.timesteps,
        reset_num_timesteps=not args.resume,
        callback=[
            CardPlayMetricsCallback(graph_dir / "cardplay_metrics.csv"),
            CardPlayCheckpointCallback(checkpoint_dir, every=args.checkpoint_every),
        ],
    )
    model.save(checkpoint_dir / "latest")
    print(f"Saved {checkpoint}")


if __name__ == "__main__":
    main()
