"""Terminal game-point rewards defined by the project plan."""

from __future__ import annotations

from typing import Sequence

NUMERIC_CONTRACTS = tuple(range(7, 14))
NOLO_REWARDS = {
    "sol": (1, 4, -6),
    "ren sol": (0, 8, -12),
    "bordlaegger": (0, 12, -18),
}


def _validate_tricks(tricks: int) -> None:
    if not isinstance(tricks, int) or not 0 <= tricks <= 13:
        raise ValueError("tricks must be an integer from 0 through 13")


def numeric_reward(bid: int, tricks: int) -> int:
    """Return the terminal reward for a numeric contract from 7 through 13."""
    if bid not in NUMERIC_CONTRACTS:
        raise ValueError("bid must be an integer from 7 through 13")
    _validate_tricks(tricks)

    bid_value = bid - 7
    if tricks >= bid:
        return bid_value + tricks - 6
    missed = bid - tricks
    locked_in_surcharge = 6 if bid == 13 else 0
    return -bid_value - locked_in_surcharge - 2 * missed


def nolo_reward(contract: str, tricks: int) -> int:
    """Return the terminal reward for Sol, Ren sol, or Bordlaegger."""
    key = contract.lower()
    if key not in NOLO_REWARDS:
        raise ValueError("contract must be Sol, Ren sol, or Bordlaegger")
    _validate_tricks(tricks)

    maximum_tricks, success_reward, failure_reward = NOLO_REWARDS[key]
    return success_reward if tricks <= maximum_tricks else failure_reward


def paskrig_rewards(tricks_by_player: Sequence[int]) -> tuple[int, ...]:
    """Return intentionally asymmetric rewards for a completed Paskrig round."""
    if len(tricks_by_player) != 4:
        raise ValueError("Paskrig requires exactly four player trick counts")
    for tricks in tricks_by_player:
        _validate_tricks(tricks)
    if sum(tricks_by_player) != 13:
        raise ValueError("completed Paskrig trick counts must sum to 13")

    fewest = min(tricks_by_player)
    most = max(tricks_by_player)
    winner_reward = most - fewest
    return tuple(
        winner_reward if tricks == fewest else -2 * (tricks - fewest)
        for tricks in tricks_by_player
    )
