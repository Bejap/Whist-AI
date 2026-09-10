"""Train an Esmakker Whist policy with historical league self-play."""

import argparse
import csv
from pathlib import Path

import matplotlib
import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from esmakker_rl_env import NUM_ACTIONS, OBS_SIZE, EsmakkerEnv


CHECKPOINT_DIR = Path("checkpoints") / "esmakker"
LEAGUE_DIR = CHECKPOINT_DIR / "league"
GRAPH_DIR = Path("graphs") / "esmakker"
METRICS_PATH = GRAPH_DIR / "training_metrics.csv"
DECISIONS_PATH = GRAPH_DIR / "bidding_decisions.csv"
PLOT_PATH = GRAPH_DIR / "training_progress.png"


class FrozenLeaguePolicy:
    """Select one frozen historical policy to control opponents each round."""

    def __init__(self, directory, device="cpu"):
        self.directory = Path(directory)
        self.device = device
        self.paths = []
        self.models = {}
        self.current_model = None
        self.failed_paths = set()
        self.refresh()

    def refresh(self):
        self.paths = sorted(self.directory.glob("snapshot_*.zip"))
        active = set(self.paths)
        self.models = {path: model for path, model in self.models.items() if path in active}

    def begin_round(self, rng):
        self.current_model = None
        if not self.paths:
            return
        path = self.paths[int(rng.integers(len(self.paths)))]
        try:
            if path not in self.models:
                model = MaskablePPO.load(path, device=self.device)
                if model.observation_space.shape != (OBS_SIZE,) or model.action_space.n != NUM_ACTIONS:
                    raise ValueError("Incompatible Esmakker observation or action space")
                self.models[path] = model
            self.current_model = self.models[path]
        except (OSError, ValueError, KeyError) as error:
            if path not in self.failed_paths:
                print(f"Skipping league snapshot {path}: {error}")
                self.failed_paths.add(path)

    def predict(self, observation, mask):
        if self.current_model is None:
            return None
        action, _ = self.current_model.predict(
            observation,
            action_masks=mask,
            deterministic=False,
        )
        return int(action)


class LeagueSnapshotCallback(BaseCallback):
    """Periodically freeze the learner for future self-play opponents."""

    def __init__(self, league, every=50_000, keep=10, verbose=0):
        super().__init__(verbose)
        self.league = league
        self.every = every
        self.keep = keep
        self.next_snapshot = every

    def _on_training_start(self):
        LEAGUE_DIR.mkdir(parents=True, exist_ok=True)
        self.next_snapshot = ((self.model.num_timesteps // self.every) + 1) * self.every
        if not self.league.paths:
            self._save_snapshot()

    def _on_step(self):
        return True

    def _on_rollout_end(self):
        if self.model.num_timesteps >= self.next_snapshot:
            self._save_snapshot()
            self.model.save(CHECKPOINT_DIR / "latest")
            self.next_snapshot = ((self.model.num_timesteps // self.every) + 1) * self.every

    def _save_snapshot(self):
        path = LEAGUE_DIR / f"snapshot_{self.model.num_timesteps:012d}.zip"
        self.model.save(path)
        snapshots = sorted(LEAGUE_DIR.glob("snapshot_*.zip"))
        for old_path in snapshots[:-self.keep]:
            old_path.unlink()
        self.league.refresh()
        print(f"League snapshot saved: {path}")


class EsmakkerMetricsCallback(BaseCallback):
    """Persist round outcomes and refresh a compact progress graph."""

    def __init__(self, report_every=500, verbose=0):
        super().__init__(verbose)
        self.report_every = report_every
        self.rows = []
        self.decisions = []

    def _on_training_start(self):
        GRAPH_DIR.mkdir(parents=True, exist_ok=True)
        if METRICS_PATH.exists():
            with METRICS_PATH.open(newline="", encoding="utf-8") as handle:
                self.rows = list(csv.DictReader(handle))
        if DECISIONS_PATH.exists():
            with DECISIONS_PATH.open(newline="", encoding="utf-8") as handle:
                self.decisions = list(csv.DictReader(handle))

    def _on_step(self):
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            decision = info.get("last_decision")
            if decision is not None:
                self.decisions.append({**decision, "episode": len(self.rows) + 1})
            settlement = info.get("settlement")
            if not done or settlement is None:
                continue
            tricks = info.get("tricks_won", [0, 0, 0, 0])
            player = int(info.get("learning_player", 0))
            payment = settlement.payments[player]
            if payment is None:
                continue
            self.rows.append({
                "episode": len(self.rows) + 1,
                "timesteps": self.num_timesteps,
                "contract": info.get("contract") or "none",
                "success": int(settlement.contract_succeeded),
                "payment": payment,
                "learning_player_tricks": tricks[player],
                "value": settlement.value,
                "underbid_penalty": info.get("underbid_penalty", 0.0),
            })
        if self.rows and len(self.rows) % self.report_every == 0:
            self._write_metrics()
        return True

    def _on_training_end(self):
        self._write_metrics()

    def _write_metrics(self):
        fieldnames = [
            "episode", "timesteps", "contract", "success", "payment",
            "learning_player_tricks", "value",
            "underbid_penalty",
        ]
        for row in self.rows:
            row.setdefault("underbid_penalty", 0.0)
        with METRICS_PATH.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.rows)

        decision_fields = [
            "episode", "phase", "action", "current_bid_before", "high_cards",
            "aces", "longest_suit", "final_contract", "success",
            "learning_player_tricks", "settlement",
        ]
        results = {str(row["episode"]): row for row in self.rows}
        enriched = []
        for decision in self.decisions:
            result = results.get(str(decision["episode"]), {})
            enriched.append({
                **decision,
                "final_contract": result.get("contract", ""),
                "success": result.get("success", ""),
                "learning_player_tricks": result.get("learning_player_tricks", ""),
                "settlement": result.get("payment", ""),
            })
        with DECISIONS_PATH.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=decision_fields)
            writer.writeheader()
            writer.writerows(enriched)

        recent = self.rows[-100:]
        episodes = np.arange(len(self.rows) - len(recent) + 1, len(self.rows) + 1)
        payments = np.asarray([float(row["payment"]) for row in recent])
        success = np.asarray([float(row["success"]) for row in recent])
        fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        axes[0].plot(episodes, payments, alpha=0.25, color="tab:blue")
        if len(payments) >= 10:
            axes[0].plot(episodes[9:], np.convolve(payments, np.ones(10) / 10, mode="valid"), color="tab:blue")
        axes[0].axhline(0, color="black", linewidth=0.8)
        axes[0].set_ylabel("Payment / settlement")
        axes[0].set_title("Esmakker training: latest 100 rounds")
        axes[1].plot(episodes, success, alpha=0.25, color="tab:green")
        if len(success) >= 10:
            axes[1].plot(episodes[9:], np.convolve(success, np.ones(10) / 10, mode="valid"), color="tab:green")
        axes[1].set_ylabel("Contract success")
        axes[1].set_xlabel("Completed rounds")
        axes[1].set_ylim(-0.05, 1.05)
        fig.tight_layout()
        fig.savefig(PLOT_PATH, dpi=140)
        plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=250_000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--opponent-device", default="cpu")
    parser.add_argument("--snapshot-every", type=int, default=50_000)
    parser.add_argument("--league-size", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = CHECKPOINT_DIR / "latest.zip"
    league = FrozenLeaguePolicy(LEAGUE_DIR, device=args.opponent_device)
    env = EsmakkerEnv(opponent_policy=league)

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

    callbacks = [
        EsmakkerMetricsCallback(),
        LeagueSnapshotCallback(
            league,
            every=args.snapshot_every,
            keep=args.league_size,
        ),
    ]
    model.learn(
        total_timesteps=args.timesteps,
        reset_num_timesteps=not args.resume,
        callback=callbacks,
    )
    model.save(CHECKPOINT_DIR / "latest")
    print(f"Saved {checkpoint}")


if __name__ == "__main__":
    main()