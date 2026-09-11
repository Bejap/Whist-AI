"""Rules engine for the Esmakker Whist variant.

The engine deliberately separates bidding, declaration and card play so a
future learning environment can model each decision type without changing game
rules.
"""

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

from whist_env import CARDS_PER_PLAYER, NUM_CARDS, NUM_PLAYERS, RANKS, SUITS, card_name

NO_TRUMP = 4
RANK_ACE = 12
RANK_KING = 11
BID_ORDER = ("7", "8", "sol", "9", "10", "ren_sol", "11", "12", "bordlaegger", "13")
NUMERIC_BIDS = {str(number): number for number in range(7, 14)}
NOLO_BIDS = {"sol", "ren_sol", "bordlaegger"}
NOLO_VALUES = {"sol": 2, "ren_sol": 4, "bordlaegger": 8}
CONTRACT_13_REWARD_SCALE = 25.0
PASKRIG = "paskrig"
PASKRIG_TRIGGER = os.getenv("PASKRIG_TRIGGER", "auto_on_all_pass")
if PASKRIG_TRIGGER not in {"auto_on_all_pass", "callable_bid"}:
    raise ValueError("PASKRIG_TRIGGER must be 'auto_on_all_pass' or 'callable_bid'.")


@dataclass(frozen=True)
class Settlement:
    contract_succeeded: bool
    value: float
    payments: tuple[float, float, float, float]


class EsmakkerGame:
    """Stateful Esmakker rules engine for exactly four players."""

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)
        self.dealer = int(self.rng.integers(NUM_PLAYERS))
        self.reset_round()

    def reset_round(self):
        deck = np.arange(NUM_CARDS)
        self.rng.shuffle(deck)
        self.hands = [sorted(deck[index * CARDS_PER_PLAYER:(index + 1) * CARDS_PER_PLAYER].tolist()) for index in range(NUM_PLAYERS)]
        self.phase = "bidding"
        self.current_player = (self.dealer + 1) % NUM_PLAYERS
        self.active_bidders = set(range(NUM_PLAYERS))
        self.current_bid = None
        self.declarer = None
        self.trump_suit = NO_TRUMP
        self.partner_suit = None
        self.partner_card = None
        self.partner_player = None
        self.partner_revealed = False
        self.special_low_ace_suit = None
        self.trick_cards = []
        self.lead_player = None
        self.tricks_won = [0] * NUM_PLAYERS
        self.round_settlement = None
        return self.public_state()

    def bid(self, player: int, contract: str):
        self._require_phase("bidding")
        self._require_current_player(player)
        if player not in self.active_bidders:
            raise ValueError("A player who passed may not bid again.")
        if contract == PASKRIG and PASKRIG_TRIGGER == "callable_bid":
            self.current_bid = PASKRIG
            self.declarer = None
            self._begin_paskrig()
            return self.public_state()
        if contract not in BID_ORDER:
            raise ValueError(f"Unknown contract: {contract}")
        if self.current_bid is not None and BID_ORDER.index(contract) <= BID_ORDER.index(self.current_bid):
            raise ValueError("A bid must be higher than the current bid.")
        self.current_bid = contract
        self.declarer = player
        self._advance_bidding_player()
        return self.public_state()

    def pass_bid(self, player: int):
        self._require_phase("bidding")
        self._require_current_player(player)
        self.active_bidders.discard(player)
        if not self.active_bidders:
            if self.current_bid is None:
                if PASKRIG_TRIGGER == "auto_on_all_pass":
                    self._begin_paskrig()
                    return self.public_state()
                self._advance_dealer(nolo=False)
                return self.reset_round()
            self._begin_declaration()
            return self.public_state()
        if len(self.active_bidders) == 1 and self.current_bid is not None:
            self._begin_declaration()
            return self.public_state()
        self._advance_bidding_player()
        return self.public_state()

    def choose_trump(self, player: int, suit: int):
        self._require_phase("choose_trump")
        if player != self.declarer:
            raise ValueError("Only declarer names trump.")
        if suit not in range(len(SUITS)):
            raise ValueError("Trump must be a suit.")
        self.trump_suit = suit
        self.phase = "choose_partner"
        return self.public_state()

    def choose_partner_suit(self, player: int, suit: int):
        self._require_phase("choose_partner")
        if player != self.declarer:
            raise ValueError("Only declarer names the partner suit.")
        if suit not in range(len(SUITS)) or suit == self.trump_suit:
            raise ValueError("Partner suit must be a non-trump suit.")
        all_aces = all(suit_index * 13 + RANK_ACE in self.hands[self.declarer] for suit_index in range(len(SUITS)))
        self.partner_suit = suit
        self.partner_card = suit * 13 + (RANK_KING if all_aces else RANK_ACE)
        self.special_low_ace_suit = suit if all_aces else None
        self.partner_player = next(
            player_index for player_index, hand in enumerate(self.hands)
            if self.partner_card in hand
        )
        self._start_tricks()
        return self.public_state()

    def play_card(self, player: int, card: int):
        self._require_phase("play")
        self._require_current_player(player)
        legal = self.legal_cards(player)
        if card not in legal:
            raise ValueError("Card is not legal in the current trick.")
        self.hands[player].remove(card)
        self.trick_cards.append((player, card))
        if len(self.trick_cards) < NUM_PLAYERS:
            self.current_player = (player + 1) % NUM_PLAYERS
            return self.public_state()

        winner = self.trick_winner(self.trick_cards)
        self.tricks_won[winner] += 1
        self.trick_cards = []
        self.lead_player = winner
        self.current_player = winner
        if sum(self.tricks_won) == CARDS_PER_PLAYER:
            self.round_settlement = self.settle()
            self.phase = "complete"
            self._advance_dealer(self.current_bid in NOLO_BIDS)
        return self.public_state()

    def legal_cards(self, player: int):
        if self.phase != "play" or player != self.current_player:
            return []
        hand = self.hands[player]
        if not self.trick_cards:
            return list(hand)
        lead_suit = self.trick_cards[0][1] // 13
        follow = [card for card in hand if card // 13 == lead_suit]
        legal = follow or list(hand)
        if lead_suit == self.partner_suit and self.partner_card in legal:
            return [self.partner_card]
        return legal

    def trick_winner(self, cards):
        winner, best_card = cards[0]
        lead_suit = best_card // 13
        for player, card in cards[1:]:
            if self._beats(card, best_card, lead_suit):
                winner, best_card = player, card
        if self.partner_card in [card for _, card in cards]:
            self.partner_revealed = True
        return winner

    def settle(self):
        if self.current_bid == PASKRIG:
            fewest = min(self.tricks_won)
            winners = [player for player, tricks in enumerate(self.tricks_won) if tricks == fewest]
            payments = [float(-(tricks - fewest)) for tricks in self.tricks_won]
            pot = -sum(payments)
            share = pot / len(winners)
            for winner in winners:
                payments[winner] = share
            return Settlement(True, pot, tuple(payments))

        declarer_tricks = self.tricks_won[self.declarer]
        if self.current_bid in NOLO_BIDS:
            limit = 1 if self.current_bid == "sol" else 0
            succeeded = declarer_tricks <= limit
            value = NOLO_VALUES[self.current_bid]
            payments = [0] * NUM_PLAYERS
            for player in range(NUM_PLAYERS):
                if player == self.declarer:
                    payments[player] = value * (NUM_PLAYERS - 1) if succeeded else -value * (NUM_PLAYERS - 1)
                else:
                    payments[player] = -value if succeeded else value
            return Settlement(succeeded, value, tuple(payments))

        team = {self.declarer, self.partner_player}
        team_tricks = sum(self.tricks_won[player] for player in team)
        target = NUMERIC_BIDS[self.current_bid]
        base = 25 if target == 13 else max(0, target - 7)
        succeeded = team_tricks >= target
        value = base + (team_tricks - 6) if succeeded and target != 13 else base
        if not succeeded:
            value = base + 2 * (target - team_tricks)
        payments = [0] * NUM_PLAYERS
        defenders = [player for player in range(NUM_PLAYERS) if player not in team]
        for index, team_player in enumerate(sorted(team)):
            defender = defenders[index]
            payments[team_player] = value if succeeded else -value
            payments[defender] = -value if succeeded else value
        return Settlement(succeeded, value, tuple(payments))

    def public_state(self):
        return {
            "phase": self.phase,
            "dealer": self.dealer,
            "current_player": self.current_player,
            "current_bid": self.current_bid,
            "declarer": self.declarer,
            "trump_suit": self.trump_suit,
            "partner_suit": self.partner_suit if self.partner_revealed else None,
            "partner_revealed": self.partner_revealed,
            "trick_cards": list(self.trick_cards),
            "tricks_won": list(self.tricks_won),
            "bordlaegger_hand_visible": self.current_bid == "bordlaegger",
            "settlement": self.round_settlement,
        }

    def player_state(self, player: int):
        state = self.public_state()
        state["hand"] = list(self.hands[player])
        state["legal_cards"] = self.legal_cards(player)
        if player == self.declarer:
            state["partner_suit"] = self.partner_suit
        if self.current_bid == "bordlaegger":
            state["declarer_hand"] = list(self.hands[self.declarer])
        return state

    def _begin_declaration(self):
        if self.current_bid in NOLO_BIDS:
            self.trump_suit = NO_TRUMP
            self._start_tricks()
        else:
            self.phase = "choose_trump"

    def _begin_paskrig(self):
        self.current_bid = PASKRIG
        self.declarer = None
        self.trump_suit = NO_TRUMP
        self.partner_suit = None
        self.partner_card = None
        self.partner_player = None
        self._start_tricks()

    def _start_tricks(self):
        self.phase = "play"
        self.current_player = (self.dealer + 1) % NUM_PLAYERS
        self.lead_player = self.current_player

    def _advance_bidding_player(self):
        for offset in range(1, NUM_PLAYERS + 1):
            candidate = (self.current_player + offset) % NUM_PLAYERS
            if candidate in self.active_bidders:
                self.current_player = candidate
                return
        self._begin_declaration()

    def _advance_dealer(self, nolo: bool):
        if not nolo:
            self.dealer = (self.dealer + 1) % NUM_PLAYERS

    def _beats(self, candidate: int, current: int, lead_suit: int):
        candidate_suit, current_suit = candidate // 13, current // 13
        if self.trump_suit != NO_TRUMP:
            if candidate_suit == self.trump_suit and current_suit != self.trump_suit:
                return True
            if current_suit == self.trump_suit and candidate_suit != self.trump_suit:
                return False
        if candidate_suit != lead_suit:
            return False
        if current_suit != lead_suit:
            return True
        return self._rank(candidate) > self._rank(current)

    def _rank(self, card: int):
        suit, rank = card // 13, card % 13
        if self.current_bid == PASKRIG or self.current_bid in NOLO_BIDS:
            return -1 if rank == RANK_ACE else rank
        if suit == self.special_low_ace_suit and rank == RANK_ACE:
            return -1
        return rank

    def _require_phase(self, phase: str):
        if self.phase != phase:
            raise RuntimeError(f"Expected phase {phase}, got {self.phase}.")

    def _require_current_player(self, player: int):
        if player != self.current_player:
            raise ValueError("It is not this player's turn.")
