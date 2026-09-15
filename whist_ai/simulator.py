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
        played_cards = tuple(
            card
            for trick in game.trick_history
            for _, card in trick
        ) + tuple(card for _, card in game.trick_cards)
        played_cards_by_player = tuple(
            (played_player, card)
            for trick in game.trick_history
            for played_player, card in trick
        ) + tuple(game.trick_cards)
        void_suits = tuple(
            (played_player, trick[0][1][0])
            for trick in game.trick_history
            for played_player, card in trick
            if card[0] != trick[0][1][0]
        )
        bid_history = tuple(
            (decision.player, decision.action)
            for decision in self.decisions
            if decision.phase == "bidding"
        )
        return {
            "player": player,
            "phase": game.phase,
            "hand": tuple(game.hands[player]),
            "current_bid": game.current_bid,
            "trick_cards": visible_tricks,
            "played_cards": played_cards,
            "played_cards_by_player": played_cards_by_player,
            "current_trick_by_player": tuple(game.trick_cards),
            "void_suits": void_suits,
            "bid_history": bid_history,
            "passed_players": tuple(sorted(game.passed_players)),
            "tricks_won": tuple(game.tricks_won),
            "dealer": game.dealer,
            "trump_suit": game.trump_suit,
            "partner_suit": game.partner_suit,
            "declarer": game.declarer,
            "partner_revealed": game.partner_player is not None,
            "partner_player": game.partner_player,
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
