"""Gymnasium adapter for training an Esmakker Whist policy with MaskablePPO."""

import os

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from esmakker_env import BID_ORDER, NUMERIC_BIDS, PASKRIG, EsmakkerGame
from whist_env import NUM_CARDS, NUM_PLAYERS


CARD_ACTIONS = NUM_CARDS
BID_START = CARD_ACTIONS
PASS_ACTION = BID_START + len(BID_ORDER)
TRUMP_START = PASS_ACTION + 1
PARTNER_START = TRUMP_START + 4
NUM_ACTIONS = PARTNER_START + 4

PHASES = ("bidding", "choose_trump", "choose_partner", "play")
OBS_CONTRACTS = BID_ORDER + (PASKRIG,)
OBS_SIZE = 52 + 52 + 4 + 12 + 5 + 4 + 5 + 4 + 4 + 4
OPENING_PASS_PENALTY = float(os.getenv("ESMAKKER_OPENING_PASS_PENALTY", "0.5"))
SETTLEMENT_REWARD_SCALE = float(os.getenv("ESMAKKER_SETTLEMENT_REWARD_SCALE", "10.0"))


class EsmakkerEnv(gym.Env):
    """Train one seat against policy-driven opponents for one round.

    The learning seat is randomized on reset. Opponents act until it is that
    seat's turn, so each Gymnasium step always represents one policy decision.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None, opponent_policy=None):
        super().__init__()
        self.render_mode = render_mode
        self.observation_space = spaces.Box(0.0, 1.0, shape=(OBS_SIZE,), dtype=np.float32)
        self.action_space = spaces.Discrete(NUM_ACTIONS)
        self.game = EsmakkerGame()
        self.opponent_policy = opponent_policy
        self.learning_player = 0
        self.done = False
        self.last_decision = None
        self.last_underbid_penalty = 0.0
        self.last_opening_pass_penalty = 0.0

    def set_opponent_policy(self, opponent_policy):
        self.opponent_policy = opponent_policy

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        game_seed = int(self.np_random.integers(2**31))
        self.game = EsmakkerGame(seed=game_seed)
        self.learning_player = int(self.np_random.integers(NUM_PLAYERS))
        self.done = False
        self.last_decision = None
        self.last_underbid_penalty = 0.0
        self.last_opening_pass_penalty = 0.0
        if self.opponent_policy is not None:
            self.opponent_policy.begin_round(self.np_random)
        self._advance_opponents()
        return self._observation(), self._info()

    def step(self, action):
        if self.done:
            raise RuntimeError("Episode is done. Call reset().")

        action = int(action)
        mask = self.action_masks()
        if action < 0 or action >= NUM_ACTIONS or not mask[action]:
            action = int(np.flatnonzero(mask)[0])
            invalid_penalty = -0.25
        else:
            invalid_penalty = 0.0

        self.last_decision = None
        self.last_opening_pass_penalty = 0.0
        self._take_action(self.learning_player, action)
        self._advance_opponents()

        terminated = self.game.phase == "complete"
        self.done = terminated
        reward = invalid_penalty - self.last_opening_pass_penalty
        if terminated:
            payment = self.game.round_settlement.payments[self.learning_player]
            reward += float(np.tanh(payment / SETTLEMENT_REWARD_SCALE))
        return self._observation(), reward, terminated, False, self._info()

    def action_masks(self, player=None):
        mask = np.zeros(NUM_ACTIONS, dtype=bool)
        if self.done:
            return mask
        game = self.game
        player = self.learning_player if player is None else player
        if player != self._decision_player():
            return mask
        if game.phase == "bidding":
            mask[PASS_ACTION] = True
            for index, bid in enumerate(BID_ORDER):
                if game.current_bid is None or BID_ORDER.index(bid) > BID_ORDER.index(game.current_bid):
                    mask[BID_START + index] = True
        elif game.phase == "choose_trump":
            mask[TRUMP_START:TRUMP_START + 4] = True
        elif game.phase == "choose_partner":
            for suit in range(4):
                if suit != game.trump_suit:
                    mask[PARTNER_START + suit] = True
        elif game.phase == "play":
            for card in game.legal_cards(player):
                mask[card] = True
        return mask

    def _take_action(self, player, action):
        game = self.game
        if player == self.learning_player and game.phase == "bidding":
            hand = game.hands[player]
            suit_lengths = [sum(card // 13 == suit for card in hand) for suit in range(4)]
            self.last_decision = {
                "phase": "bidding",
                "action": "pass" if action == PASS_ACTION else BID_ORDER[action - BID_START],
                "current_bid_before": game.current_bid or "none",
                "high_cards": sum(card % 13 >= 10 for card in hand),
                "aces": sum(card % 13 == 12 for card in hand),
                "longest_suit": max(suit_lengths),
                "opening_pass_penalty": (
                    OPENING_PASS_PENALTY if action == PASS_ACTION and game.current_bid is None else 0.0
                ),
            }
            self.last_opening_pass_penalty = self.last_decision["opening_pass_penalty"]
        if game.phase == "bidding":
            if action == PASS_ACTION:
                game.pass_bid(player)
            else:
                game.bid(player, BID_ORDER[action - BID_START])
        elif game.phase == "choose_trump":
            game.choose_trump(player, action - TRUMP_START)
        elif game.phase == "choose_partner":
            game.choose_partner_suit(player, action - PARTNER_START)
        else:
            game.play_card(player, action)

    def _advance_opponents(self):
        while self.game.phase != "complete" and self._decision_player() != self.learning_player:
            player = self._decision_player()
            mask = self.action_masks(player)
            action = None
            if self.opponent_policy is not None:
                action = self.opponent_policy.predict(self._observation(player), mask)
            if action is None or action < 0 or action >= NUM_ACTIONS or not mask[action]:
                action = self._rule_action(player)
            self._take_action(player, action)

    def _decision_player(self):
        if self.game.phase in {"choose_trump", "choose_partner"}:
            return self.game.declarer
        return self.game.current_player

    def _observation(self, player=None):
        game = self.game
        player = self.learning_player if player is None else player
        observation = np.zeros(OBS_SIZE, dtype=np.float32)
        index = 0
        for card in game.hands[player]:
            observation[index + card] = 1.0
        index += 52
        for _, card in game.trick_cards:
            observation[index + card] = 1.0
        index += 52
        if game.phase in PHASES:
            observation[index + PHASES.index(game.phase)] = 1.0
        index += 4
        observation[index + (OBS_CONTRACTS.index(game.current_bid) + 1 if game.current_bid else 0)] = 1.0
        index += 12
        observation[index + game.trump_suit] = 1.0
        index += 5
        observation[index + player] = 1.0
        index += 4
        observation[index + (game.declarer if game.declarer is not None else 4)] = 1.0
        index += 5
        observation[index:index + 4] = np.asarray(game.tricks_won, dtype=np.float32) / 13.0
        index += 4
        for player, _ in game.trick_cards:
            observation[index + player] = 1.0
        index += 4
        if game.partner_revealed and game.partner_player is not None:
            observation[index + game.partner_player] = 1.0
        return observation

    def _info(self):
        return {
            "phase": self.game.phase,
            "learning_player": self.learning_player,
            "declarer": self.game.declarer,
            "action_mask": self.action_masks(),
            "settlement": self.game.round_settlement,
            "contract": self.game.current_bid,
            "tricks_won": list(self.game.tricks_won),
            "last_decision": self.last_decision,
            "underbid_penalty": self.last_underbid_penalty,
            "opening_pass_penalty": self.last_opening_pass_penalty,
        }

    def _hand_strength(self, player):
        return sum(card % 13 >= 10 for card in self.game.hands[player])

    def _best_suit(self, player):
        hand = self.game.hands[player]
        return max(range(4), key=lambda suit: sum(card // 13 == suit and card % 13 >= 9 for card in hand))

    def _bot_card(self, player):
        legal = self.game.legal_cards(player)
        return max(legal, key=lambda card: card % 13)

    def _rule_action(self, player):
        game = self.game
        if game.phase == "bidding":
            if game.current_bid is None and self._hand_strength(player) >= 10:
                return BID_START
            return PASS_ACTION
        if game.phase == "choose_trump":
            return TRUMP_START + self._best_suit(player)
        if game.phase == "choose_partner":
            suit = next(suit for suit in range(4) if suit != game.trump_suit)
            return PARTNER_START + suit
        return self._bot_card(player)