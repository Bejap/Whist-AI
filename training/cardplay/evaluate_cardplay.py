"""Benchmark a trained card-play specialist across numeric contracts."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO

from training.cardplay.esmakker_cardplay_env import EsmakkerCardPlayEnv


CONTRACTS = tuple(str(number) for number in range(7, 14))


TABLES = ("random", "rules", "learned")


def evaluate_checkpoint(
    checkpoint, contracts, episodes, seed, device, max_steps, opponent_mode="rules",
    opponent_checkpoint=None,
):
    """Evaluate one checkpoint with deterministic actions and fixed seeds."""
    model = MaskablePPO.load(checkpoint, device=device)
    opponent_model = None
    if opponent_mode == "learned":
        if opponent_checkpoint is None:
            raise ValueError("opponent_checkpoint is required for learned opponents")
        opponent_model = MaskablePPO.load(opponent_checkpoint, device=device)
    rows = []
    for contract_index, contract in enumerate(contracts):
        env = EsmakkerCardPlayEnv(
            contract=contract,
            opponent_mode=opponent_mode,
            opponent_model=opponent_model,
        )
        for episode in range(episodes):
            episode_seed = seed + contract_index * episodes + episode
            observation, _ = env.reset(seed=episode_seed)
            terminated = False
            steps = 0
            while not terminated and steps < max_steps:
                action, _ = model.predict(
                    observation,
                    action_masks=env.action_masks(),
                    deterministic=True,
                )
                observation, _, terminated, _, info = env.step(int(action))
                steps += 1
            settlement = info["settlement"]
            payment = settlement.payments[info["learning_player"]] if settlement else 0.0
            rows.append({
                "contract": contract,
                "table": opponent_mode,
                "episode": episode + 1,
                "seed": episode_seed,
                "completed": int(terminated and settlement is not None),
                "contract_success": int(settlement.contract_succeeded) if settlement else 0,
                "success": int(payment > 0) if settlement else 0,
                "payment": payment,
                "team_tricks": info["team_tricks"],
                "steps": steps,
            })
    return rows


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row.get("table", "rules"), row["contract"])].append(row)
    summary = []
    for (table, contract), contract_rows in grouped.items():
        completed = [row for row in contract_rows if row["completed"]]
        payments = np.asarray([float(row["payment"]) for row in completed], dtype=float)
        tricks = np.asarray([float(row["team_tricks"]) for row in completed], dtype=float)
        successes = np.asarray([float(row["success"]) for row in completed], dtype=float)
        summary.append({
            "table": table,
            "contract": contract,
            "episodes": len(contract_rows),
            "completed": len(completed),
            "completion_rate": len(completed) / len(contract_rows),
            "success_rate": float(successes.mean()) if len(successes) else 0.0,
            "average_payment": float(payments.mean()) if len(payments) else 0.0,
            "average_team_tricks": float(tricks.mean()) if len(tricks) else 0.0,
        })
    summary.sort(key=lambda row: (row["table"], int(row["contract"])))
    return summary


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--contract", choices=list(CONTRACTS) + ["all"], default="all")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=50_000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--opponents", choices=list(TABLES) + ["all"], default="all")
    parser.add_argument("--opponent-checkpoint", type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.episodes <= 0 or args.max_steps <= 0:
        raise ValueError("--episodes and --max-steps must be positive")
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    contracts = CONTRACTS if args.contract == "all" else (args.contract,)
    output_dir = args.output_dir or Path("graphs") / checkpoint.parent.name
    opponent_modes = TABLES if args.opponents == "all" else (args.opponents,)
    if "learned" in opponent_modes and args.opponent_checkpoint is None:
        raise ValueError("--opponent-checkpoint is required when benchmarking learned opponents")
    rows = []
    for table_index, opponent_mode in enumerate(opponent_modes):
        rows.extend(evaluate_checkpoint(
            checkpoint,
            contracts,
            args.episodes,
            seed=args.seed + table_index * len(contracts) * args.episodes,
            device=args.device,
            max_steps=args.max_steps,
            opponent_mode=opponent_mode,
            opponent_checkpoint=args.opponent_checkpoint,
        ))
    summary = summarize(rows)
    write_csv(
        output_dir / "cardplay_benchmark_games.csv",
        rows,
        [
            "table", "contract", "episode", "seed", "completed", "success",
            "contract_success", "payment", "team_tricks", "steps",
        ],
    )
    write_csv(
        output_dir / "cardplay_benchmark_summary.csv",
        summary,
        [
            "table", "contract", "episodes", "completed", "completion_rate", "success_rate",
            "average_payment", "average_team_tricks",
        ],
    )
    for result in summary:
        print(
            f"Table {result['table']} | contract {result['contract']}: "
            f"success {result['success_rate']:.1%} | "
            f"team tricks {result['average_team_tricks']:.3f} | "
            f"payment {result['average_payment']:+.3f} | "
            f"completed {result['completed']}/{result['episodes']}"
        )


if __name__ == "__main__":
    main()
