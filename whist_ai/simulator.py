"""Phase-aware controller and perspective observations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from .engine import Card, WhistGame
from .policies import Policy


@dataclass(frozen=True)
class Decision:
    player: int
    phase: str
    action: Any


class GameController:
    def __init__(self, game: WhistGame, policies: Sequence[Policy]):
        if len(policies) != 4:
            raise ValueError("exactly four seat policies are required")
        self.game = game
        self.policies = tuple(policies)
        self.decisions: list[Decision] = []

    def observation(self, player: int) -> dict[str, Any]:
        game = self.game
        visible_tricks = tuple(game.trick_cards)
        return {
            "player": player,
            "phase": game.phase,
            "hand": tuple(game.hands[player]),
            "current_bid": game.current_bid,
            "trick_cards": visible_tricks,
            "tricks_won": tuple(game.tricks_won),
            "trump_suit": game.trump_suit,
            "declarer": game.declarer,
            "partner_revealed": False,
            "partner_player": None,
        }

    def step(self) -> Decision:
        if self.game.phase == "complete":
            raise RuntimeError("game is complete")
        player = self.game.current_player
        policy = self.policies[player]
        observation = self.observation(player)
        phase = self.game.phase
        if phase == "bidding":
            legal = self.game.legal_bids()
            action = policy.choose_bid(observation, legal)
            if action == "pass":
                self.game.pass_bid(player)
            else:
                self.game.bid(player, action)
        elif phase == "choose_trump":
            legal = self.game.legal_trump_suits(player)
            action = policy.choose_trump(observation, legal)
            self.game.choose_trump(player, action)
        elif phase == "choose_partner":
            legal = self.game.legal_partner_suits(player)
            action = policy.choose_partner(observation, legal)
            self.game.choose_partner_suit(player, action)
        elif phase == "play":
            legal = self.game.legal_cards(player)
            action = policy.choose_card(observation, legal)
            self.game.play_card(player, action)
        else:
            raise RuntimeError(f"unknown phase {phase}")
        decision = Decision(player, phase, action)
        self.decisions.append(decision)
        return decision

    def run(self, max_decisions: int = 300) -> tuple[int, ...]:
        for _ in range(max_decisions):
            if self.game.phase == "complete":
                return tuple(self.game.terminal_reward(player) for player in range(4))
            self.step()
        raise RuntimeError("game exceeded decision safety limit")
