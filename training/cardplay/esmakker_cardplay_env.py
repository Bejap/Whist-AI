"""Fixed-contract card-play curriculum for Esmakker."""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from esmakker_env import BID_ORDER, CONTRACT_13_REWARD_SCALE, NUMERIC_BIDS, EsmakkerGame
from esmakker_rl_env import OBS_CONTRACTS, OBS_SIZE
from whist_env import NUM_CARDS, NUM_PLAYERS

CARD_PLAY_REWARD_SCALE = 50.0
TRICK_REWARD = 0.1
CARD_PLAY_OBS_SIZE = OBS_SIZE + NUM_CARDS + (NUM_PLAYERS * NUM_CARDS) + (NUM_PLAYERS * 4)
CONTRACT_SAMPLING_ORDER = ("9", "8", "7", "10", "11", "12", "13")
CONTRACT_SAMPLING_WEIGHTS = np.asarray(
    [1.0 / rank for rank in range(1, len(CONTRACT_SAMPLING_ORDER) + 1)],
    dtype=np.float64,
)
CONTRACT_SAMPLING_PROBABILITIES = CONTRACT_SAMPLING_WEIGHTS / CONTRACT_SAMPLING_WEIGHTS.sum()


class EsmakkerCardPlayEnv(gym.Env):
    """Train card play for one fixed contract without any bidding decisions."""

    metadata = {"render_modes": []}

    def __init__(self, contract="7", render_mode=None):
        super().__init__()
        if contract != "all" and (contract not in BID_ORDER or contract not in NUMERIC_BIDS):
            raise ValueError("Card-play curriculum supports numeric contracts 7 through 13 or 'all'.")
        self.contract_mode = contract
        self.contract = contract
        self.render_mode = render_mode
        self.observation_space = spaces.Box(
            0.0, 1.0, shape=(CARD_PLAY_OBS_SIZE,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(NUM_CARDS)
        self.game = EsmakkerGame()
        self.learning_player = 0
        self.done = False
        self.previous_team_tricks = 0
        self.void_suits = np.zeros((NUM_PLAYERS, 4), dtype=bool)
        self.played_by = np.zeros((NUM_PLAYERS, NUM_CARDS), dtype=bool)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        game_seed = int(self.np_random.integers(2**31))
        self.game = EsmakkerGame(seed=game_seed)
        self.learning_player = int(self.np_random.integers(NUM_PLAYERS))
        self.game.declarer = int(self.np_random.integers(NUM_PLAYERS))
        if self.contract_mode == "all":
            self.contract = str(self.np_random.choice(
                CONTRACT_SAMPLING_ORDER,
                p=CONTRACT_SAMPLING_PROBABILITIES,
            ))
        self.game.current_bid = self.contract
        self.game.trump_suit = int(self.np_random.integers(4))
        partner_suit = (self.game.trump_suit + int(self.np_random.integers(1, 4))) % 4
        self.game.partner_suit = partner_suit
        self.game.partner_card = partner_suit * 13 + 12
        self.game.partner_player = next(
            player for player, hand in enumerate(self.game.hands)
            if self.game.partner_card in hand
        )
        self.game._start_tricks()
        self.done = False
        self.previous_team_tricks = 0
        self.void_suits.fill(False)
        self.played_by.fill(False)
        self._advance_opponents()
        return self._observation(), self._info()

    def step(self, action):
        if self.done:
            raise RuntimeError("Episode is done. Call reset().")
        mask = self.action_masks()
        action = int(action)
        invalid_penalty = 0.0
        if action < 0 or action >= NUM_CARDS or not mask[action]:
            action = int(np.flatnonzero(mask)[0])
            invalid_penalty = -0.25
        before = self._team_tricks()
        self._play_card(self.learning_player, action)
        self._advance_opponents()
        after = self._team_tricks()
        trick_reward = TRICK_REWARD * (after - before)
        terminated = self.game.phase == "complete"
        self.done = terminated
        reward = invalid_penalty + trick_reward
        if terminated:
            payment = self.game.round_settlement.payments[self.learning_player]
            scale = (
                CONTRACT_13_REWARD_SCALE
                if self.contract == "13"
                else CARD_PLAY_REWARD_SCALE
            )
            reward += payment / scale
        return self._observation(), reward, terminated, False, self._info()

    def action_masks(self, player=None):
        mask = np.zeros(NUM_CARDS, dtype=bool)
        if self.done:
            return mask
        player = self.learning_player if player is None else player
        if player == self.game.current_player and self.game.phase == "play":
            mask[self.game.legal_cards(player)] = True
        return mask

    def _advance_opponents(self):
        while self.game.phase != "complete" and self.game.current_player != self.learning_player:
            player = self.game.current_player
            legal = self.game.legal_cards(player)
            self._play_card(player, max(legal, key=lambda card: card % 13))

    def _play_card(self, player, card):
        if self.game.trick_cards:
            lead_suit = self.game.trick_cards[0][1] // 13
            if not any(held_card // 13 == lead_suit for held_card in self.game.hands[player]):
                self.void_suits[player, lead_suit] = True
        self.played_by[player, card] = True
        self.game.play_card(player, card)

    def _team_tricks(self):
        team = {self.learning_player, self.game.partner_player}
        return sum(self.game.tricks_won[player] for player in team)

    def _observation(self):
        observation = np.zeros(CARD_PLAY_OBS_SIZE, dtype=np.float32)
        index = 0
        for card in self.game.hands[self.learning_player]:
            observation[index + card] = 1.0
        index += 52
        for _, card in self.game.trick_cards:
            observation[index + card] = 1.0
        index += 52
        observation[index + 3] = 1.0
        index += 4
        observation[index + OBS_CONTRACTS.index(self.contract) + 1] = 1.0
        index += 12
        observation[index + self.game.trump_suit] = 1.0
        index += 5
        observation[index + self.learning_player] = 1.0
        index += 4
        observation[index + self.game.declarer] = 1.0
        index += 5
        observation[index:index + 4] = np.asarray(self.game.tricks_won, dtype=np.float32) / 13.0
        index += 4
        for player, _ in self.game.trick_cards:
            observation[index + player] = 1.0
        index += 4
        if self.game.partner_revealed:
            observation[index + self.game.partner_player] = 1.0
        index += 4
        cards_in_hands = {card for hand in self.game.hands for card in hand}
        for card in range(NUM_CARDS):
            if card not in cards_in_hands:
                observation[index + card] = 1.0
        index += NUM_CARDS
        observation[index:index + NUM_PLAYERS * NUM_CARDS] = self.played_by.reshape(-1).astype(np.float32)
        index += NUM_PLAYERS * NUM_CARDS
        observation[index:index + NUM_PLAYERS * 4] = self.void_suits.reshape(-1).astype(np.float32)
        return observation

    def _info(self):
        return {
            "contract": self.contract,
            "learning_player": self.learning_player,
            "declarer": self.game.declarer,
            "settlement": self.game.round_settlement,
            "tricks_won": list(self.game.tricks_won),
            "team_tricks": self._team_tricks(),
        }
