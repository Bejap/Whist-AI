"""Gymnasium adapter for training an Esmakker Whist policy with MaskablePPO."""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from esmakker_env import BID_ORDER, EsmakkerGame, NOLO_BIDS
from whist_env import NUM_CARDS, NUM_PLAYERS


CARD_ACTIONS = NUM_CARDS
BID_START = CARD_ACTIONS
PASS_ACTION = BID_START + len(BID_ORDER)
TRUMP_START = PASS_ACTION + 1
PARTNER_START = TRUMP_START + 4
NUM_ACTIONS = PARTNER_START + 4

PHASES = ("bidding", "choose_trump", "choose_partner", "play")
OBS_SIZE = 52 + 52 + 4 + 11 + 5 + 4 + 5 + 4 + 4 + 4


class EsmakkerEnv(gym.Env):
    """Train one seat against legal rule-based opponents for one round.

    The learning seat is randomized on reset. Opponents act until it is that
    seat's turn, so each Gymnasium step always represents one policy decision.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.observation_space = spaces.Box(0.0, 1.0, shape=(OBS_SIZE,), dtype=np.float32)
        self.action_space = spaces.Discrete(NUM_ACTIONS)
        self.game = EsmakkerGame()
        self.learning_player = 0
        self.done = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        game_seed = int(self.np_random.integers(2**31))
        self.game = EsmakkerGame(seed=game_seed)
        self.learning_player = int(self.np_random.integers(NUM_PLAYERS))
        self.done = False
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

        self._take_action(self.learning_player, action)
        self._advance_opponents()

        terminated = self.game.phase == "complete"
        self.done = terminated
        reward = invalid_penalty
        if terminated:
            reward += self.game.round_settlement.payments[self.learning_player] / 10.0
        return self._observation(), reward, terminated, False, self._info()

    def action_masks(self):
        mask = np.zeros(NUM_ACTIONS, dtype=bool)
        if self.done:
            return mask
        game = self.game
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
            for card in game.legal_cards(self.learning_player):
                mask[card] = True
        return mask

    def _take_action(self, player, action):
        game = self.game
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
            game = self.game
            player = self._decision_player()
            if game.phase == "bidding":
                if game.current_bid is None and self._hand_strength(player) >= 10:
                    game.bid(player, "7")
                else:
                    game.pass_bid(player)
            elif game.phase == "choose_trump":
                game.choose_trump(player, self._best_suit(player))
            elif game.phase == "choose_partner":
                trump = game.trump_suit
                partner_suit = next(suit for suit in range(4) if suit != trump)
                game.choose_partner_suit(player, partner_suit)
            else:
                game.play_card(player, self._bot_card(player))

    def _decision_player(self):
        if self.game.phase in {"choose_trump", "choose_partner"}:
            return self.game.declarer
        return self.game.current_player

    def _observation(self):
        game = self.game
        observation = np.zeros(OBS_SIZE, dtype=np.float32)
        index = 0
        for card in game.hands[self.learning_player]:
            observation[index + card] = 1.0
        index += 52
        for _, card in game.trick_cards:
            observation[index + card] = 1.0
        index += 52
        if game.phase in PHASES:
            observation[index + PHASES.index(game.phase)] = 1.0
        index += 4
        observation[index + (BID_ORDER.index(game.current_bid) + 1 if game.current_bid else 0)] = 1.0
        index += 11
        observation[index + game.trump_suit] = 1.0
        index += 5
        observation[index + self.learning_player] = 1.0
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
            "action_mask": self.action_masks(),
            "settlement": self.game.round_settlement,
        }

    def _hand_strength(self, player):
        return sum(card % 13 >= 10 for card in self.game.hands[player])

    def _best_suit(self, player):
        hand = self.game.hands[player]
        return max(range(4), key=lambda suit: sum(card // 13 == suit and card % 13 >= 9 for card in hand))

    def _bot_card(self, player):
        legal = self.game.legal_cards(player)
        return max(legal, key=lambda card: card % 13)