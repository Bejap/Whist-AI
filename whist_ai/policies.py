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
    """Small deterministic baseline that is legal and easy to benchmark."""

    def choose_bid(self, observation, legal_bids):
        hand = observation["hand"]
        high_cards = sum(rank >= 10 for _, rank in hand)
        current_bid = observation["current_bid"]
        numeric = [bid for bid in legal_bids if isinstance(bid, int)]
        if current_bid is None and high_cards >= 10 and numeric:
            return 7
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
        hand = observation["hand"]
        high_cards = sum(rank >= 10 for _, rank in hand)
        numeric = [bid for bid in legal_bids if isinstance(bid, int)]
        if observation["current_bid"] is None and high_cards >= 11 and numeric:
            return 7
        return "pass" if "pass" in legal_bids else legal_bids[0]
