"""Baseline policies implementing the phase-separated controller interface."""

from __future__ import annotations

import random
from typing import Any

from .engine import BID_ORDER, Card, WhistGame


class Policy:
    """Interface shared by random, rule, search, and learned policies."""

    def choose_bid(self, observation: dict[str, Any], legal_bids: tuple[Any, ...]) -> Any:
        raise NotImplementedError

    def choose_trump(self, observation: dict[str, Any], legal_suits: tuple[int, ...]) -> int:
        raise NotImplementedError

    def choose_partner(self, observation: dict[str, Any], legal_suits: tuple[int, ...]) -> int:
        raise NotImplementedError

    def choose_card(self, observation: dict[str, Any], legal_cards: tuple[Card, ...]) -> Card:
        raise NotImplementedError


class RandomPolicy(Policy):
    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)

    def choose_bid(self, observation, legal_bids):
        return self.rng.choice(legal_bids)

    def choose_trump(self, observation, legal_suits):
        return self.rng.choice(legal_suits)

    def choose_partner(self, observation, legal_suits):
        return self.rng.choice(legal_suits)

    def choose_card(self, observation, legal_cards):
        return self.rng.choice(legal_cards)


class RulePolicy(Policy):
    """Deterministic competitive baseline with a graded bidding strategy."""

    def __init__(self, style: str = "balanced") -> None:
        if style not in {"conservative", "balanced", "aggressive"}:
            raise ValueError("style must be conservative, balanced, or aggressive")
        self.style = style

    def choose_bid(self, observation, legal_bids):
        hand = observation["hand"]
        points = sum(max(0, rank - 8) for _, rank in hand)
        aces = sum(rank == 12 for _, rank in hand)
        longest_suit = max(sum(card[0] == suit for card in hand) for suit in range(4))
        target = 7 + max(0, min(6, (points + longest_suit - 13) // 2))
        if self.style == "conservative":
            target = max(7, target - 1)
        elif self.style == "aggressive":
            target = min(13, target + 1)

        # A low-ace hand is a plausible Sol candidate when it is still legal.
        if aces <= 1 and points <= 9 and "sol" in legal_bids:
            return "sol"
        numeric = [bid for bid in legal_bids if isinstance(bid, int) and bid <= target]
        if numeric:
            return max(numeric)
        return "pass" if "pass" in legal_bids else legal_bids[0]

    def choose_trump(self, observation, legal_suits):
        hand = observation["hand"]
        return max(legal_suits, key=lambda suit: sum(card[0] == suit for card in hand))

    def choose_partner(self, observation, legal_suits):
        hand = observation["hand"]
        return max(legal_suits, key=lambda suit: sum(card[0] == suit and card[1] >= 10 for card in hand))

    def choose_card(self, observation, legal_cards):
        return max(legal_cards, key=lambda card: card[1])


class ConservativePolicy(RulePolicy):
    """Rule baseline that only bids when the hand has substantial strength."""

    def choose_bid(self, observation, legal_bids):
        return super().choose_bid(observation, legal_bids)
