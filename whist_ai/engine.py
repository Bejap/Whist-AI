"""Deterministic Esmakker game engine used by policies and environments."""

from __future__ import annotations

from dataclasses import dataclass, field
import random
from typing import Optional

from .rewards import nolo_reward, numeric_reward, paskrig_rewards

Card = tuple[int, int]
NUMERIC_BIDS = tuple(range(7, 14))
BID_ORDER = (7, 8, "sol", 9, 10, "ren sol", 11, 12, "bordlaegger", 13)
NOLO_BIDS = {"sol", "ren sol", "bordlaegger"}


class GameStateError(ValueError):
    """Raised when a policy attempts an illegal state transition."""


@dataclass
class WhistGame:
    seed: Optional[int] = None
    dealer: int = 0
    rng: random.Random = field(init=False, repr=False)
    phase: str = field(default="bidding", init=False)
    current_player: int = field(default=1, init=False)
    current_bid: int | str | None = field(default=None, init=False)
    declarer: int | None = field(default=None, init=False)
    trump_suit: int | None = field(default=None, init=False)
    partner_suit: int | None = field(default=None, init=False)
    partner_card: Card | None = field(default=None, init=False)
    partner_player: int | None = field(default=None, init=False)
    hands: list[list[Card]] = field(default_factory=list, init=False)
    passed_players: set[int] = field(default_factory=set, init=False)
    trick_cards: list[tuple[int, Card]] = field(default_factory=list, init=False)
    tricks_won: list[int] = field(default_factory=lambda: [0] * 4, init=False)
    trick_history: list[tuple[tuple[int, Card], ...]] = field(default_factory=list, init=False)
    winner: int | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)
        deck = [(suit, rank) for suit in range(4) for rank in range(13)]
        self.rng.shuffle(deck)
        self.hands = [sorted(deck[index:index + 13]) for index in range(0, 52, 13)]
        self.current_player = (self.dealer + 1) % 4

    @property
    def is_numeric(self) -> bool:
        return isinstance(self.current_bid, int)

    @property
    def active_players(self) -> tuple[int, ...]:
        return tuple(player for player in range(4) if player not in self.passed_players)

    def legal_bids(self) -> tuple[int | str | None, ...]:
        self._require_phase("bidding")
        legal = ["pass"]
        current_index = -1 if self.current_bid is None else BID_ORDER.index(self.current_bid)
        legal.extend(bid for index, bid in enumerate(BID_ORDER) if index > current_index)
        return tuple(legal)

    def bid(self, player: int, bid: int | str) -> None:
        self._require_turn(player)
        self._require_phase("bidding")
        bid = self._normalize_bid(bid)
        current_index = -1 if self.current_bid is None else BID_ORDER.index(self.current_bid)
        if BID_ORDER.index(bid) <= current_index:
            raise GameStateError("bid must be higher than the current bid")
        self.current_bid = bid
        self.declarer = player
        self.passed_players.discard(player)
        self._next_bidder()

    def pass_bid(self, player: int) -> None:
        self._require_turn(player)
        self._require_phase("bidding")
        self.passed_players.add(player)
        if self.current_bid is None and len(self.passed_players) == 4:
            self._start_paskrig()
            return
        if self.current_bid is not None and len(self.passed_players) == 3:
            self._start_contract()
            return
        self._next_bidder()

    def legal_trump_suits(self, player: int) -> tuple[int, ...]:
        self._require_turn(player)
        self._require_phase("choose_trump")
        return (0, 1, 2, 3)

    def choose_trump(self, player: int, suit: int) -> None:
        self._require_turn(player)
        self._require_phase("choose_trump")
        if suit not in range(4):
            raise GameStateError("trump suit must be 0 through 3")
        self.trump_suit = suit
        self.phase = "choose_partner"

    def legal_partner_suits(self, player: int) -> tuple[int, ...]:
        self._require_turn(player)
        self._require_phase("choose_partner")
        return tuple(suit for suit in range(4) if suit != self.trump_suit)

    def choose_partner_suit(self, player: int, suit: int) -> None:
        self._require_turn(player)
        self._require_phase("choose_partner")
        if suit not in self.legal_partner_suits(player):
            raise GameStateError("partner suit must differ from trump")
        self.partner_suit = suit
        partner_rank = 11 if self._holds_all_aces(self.declarer) else 12
        self.partner_card = (suit, partner_rank)
        self.partner_player = next(
            seat for seat, hand in enumerate(self.hands) if self.partner_card in hand
        )
        self._start_tricks(self.declarer)

    def legal_cards(self, player: int) -> tuple[Card, ...]:
        self._require_turn(player)
        self._require_phase("play")
        hand = self.hands[player]
        if not self.trick_cards:
            return tuple(hand)
        lead_suit = self.trick_cards[0][1][0]
        suited = tuple(card for card in hand if card[0] == lead_suit)
        return suited or tuple(hand)

    def play_card(self, player: int, card: Card) -> None:
        self._require_turn(player)
        legal = self.legal_cards(player)
        if card not in legal:
            raise GameStateError("card is not legal in this position")
        self.hands[player].remove(card)
        self.trick_cards.append((player, card))
        if len(self.trick_cards) < 4:
            self.current_player = (self.current_player + 1) % 4
            return
        winner = self._trick_winner()
        self.tricks_won[winner] += 1
        self.trick_history.append(tuple(self.trick_cards))
        self.trick_cards.clear()
        if self._nolo_failed() or sum(self.tricks_won) == 13:
            self.phase = "complete"
            self.winner = winner
            return
        self.current_player = winner

    def terminal_reward(self, player: int) -> int:
        if self.phase != "complete":
            raise GameStateError("terminal reward is only available after completion")
        if self.current_bid == "paskrig":
            return paskrig_rewards(self.tricks_won)[player]
        if self.is_numeric:
            team = {self.declarer, self.partner_player}
            result = numeric_reward(self.current_bid, sum(self.tricks_won[seat] for seat in team))
            return result if player in team else -result
        if player == self.declarer:
            return nolo_reward(self.current_bid, self.tricks_won[player])
        declarer_result = nolo_reward(self.current_bid, self.tricks_won[self.declarer])
        return -declarer_result

    def _start_contract(self) -> None:
        if self.current_bid in NOLO_BIDS:
            self._start_tricks(self.declarer)
        else:
            self.phase = "choose_trump"
            self.current_player = self.declarer

    def _start_paskrig(self) -> None:
        self.current_bid = "paskrig"
        self.declarer = None
        self.trump_suit = None
        self.partner_suit = None
        self.partner_card = None
        self.partner_player = None
        self._start_tricks((self.dealer + 1) % 4)

    def _start_tricks(self, leader: int) -> None:
        self.phase = "play"
        self.current_player = leader

    def _next_bidder(self) -> None:
        for offset in range(1, 5):
            candidate = (self.current_player + offset) % 4
            if candidate not in self.passed_players:
                self.current_player = candidate
                return
        raise GameStateError("no bidder remains")

    def _trick_winner(self) -> int:
        led_suit = self.trick_cards[0][1][0]
        nolo = self.current_bid in NOLO_BIDS or self.current_bid == "paskrig"

        def rank(item: tuple[int, Card]) -> tuple[int, int]:
            player, (suit, card_rank) = item
            value = 0 if nolo and card_rank == 12 else card_rank + 1
            trump = 1 if self.trump_suit is not None and suit == self.trump_suit else 0
            return (trump, value) if suit == led_suit or trump else (-1, -1)

        return max(self.trick_cards, key=rank)[0]

    def _nolo_failed(self) -> bool:
        if self.current_bid not in NOLO_BIDS:
            return False
        maximum, _, _ = {
            "sol": (1, 4, -6),
            "ren sol": (0, 8, -12),
            "bordlaegger": (0, 12, -18),
        }[self.current_bid]
        return self.tricks_won[self.declarer] > maximum

    def _holds_all_aces(self, player: int | None) -> bool:
        return player is not None and all((suit, 12) in self.hands[player] for suit in range(4))

    @staticmethod
    def _normalize_bid(bid: int | str) -> int | str:
        if isinstance(bid, int) and bid in NUMERIC_BIDS:
            return bid
        normalized = str(bid).lower().replace("æ", "ae")
        if normalized == "bordlaegger":
            return "bordlaegger"
        if normalized in NOLO_BIDS:
            return normalized
        raise GameStateError("unknown contract")

    def _require_phase(self, phase: str) -> None:
        if self.phase != phase:
            raise GameStateError(f"expected phase {phase}, got {self.phase}")

    def _require_turn(self, player: int) -> None:
        if player != self.current_player:
            raise GameStateError("player is not the current actor")
        if player not in range(4):
            raise GameStateError("player must be 0 through 3")
