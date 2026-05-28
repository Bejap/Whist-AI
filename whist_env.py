"""Whist card game environment compatible with Gymnasium."""

import gymnasium as gym
import numpy as np
from gymnasium import spaces


# Card encoding: 0-51, suit = card // 13, rank = card % 13
SUITS = ["Clubs", "Diamonds", "Hearts", "Spades"]
RANKS = [
    "2", "3", "4", "5", "6", "7", "8", "9", "10",
    "Jack", "Queen", "King", "Ace",
]
NUM_CARDS = 52
NUM_PLAYERS = 4
CARDS_PER_PLAYER = 13
NUM_TRICKS = 13

# Teams: team 0 = players 0, 2; team 1 = players 1, 3
TEAMS = {0: 0, 1: 1, 2: 0, 3: 1}


def card_name(card_id: int) -> str:
    """Return a human-readable card name."""
    return f"{RANKS[card_id % 13]} of {SUITS[card_id // 13]}"


class WhistEnv(gym.Env):
    """Gymnasium environment for 4-player Whist.

    Observation (length 224):
        - own hand:       52 bits (one-hot)
        - played cards:   52 bits (cards already used in previous tricks)
        - current trick:  52 bits (cards on the table this trick, up to 3)
        - trump suit:      4 bits (one-hot)
        - team tricks:    64 floats (team0 tricks / 13, team1 tricks / 13,
                          repeated 32× for alignment – simplified to 2 floats
                          padded to keep space compact)
    Simplified observation = 52 + 52 + 52 + 4 + 2 = 162

    Action space: Discrete(52), masked to valid cards in hand.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(162,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(NUM_CARDS)

        # State variables
        self.hands = [[] for _ in range(NUM_PLAYERS)]
        self.trump_suit = 0
        self.current_player = 0
        self.trick_cards = []  # list of (player, card) for current trick
        self.lead_player = 0
        self.played_cards = np.zeros(NUM_CARDS, dtype=np.float32)
        self.team_tricks = [0, 0]
        self.tricks_played = 0
        self.done = False

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        deck = list(range(NUM_CARDS))
        self.np_random.shuffle(deck)

        self.hands = [
            sorted(deck[i * CARDS_PER_PLAYER: (i + 1) * CARDS_PER_PLAYER])
            for i in range(NUM_PLAYERS)
        ]
        self.trump_suit = self.np_random.integers(0, 4)
        self.current_player = 0
        self.lead_player = 0
        self.trick_cards = []
        self.played_cards = np.zeros(NUM_CARDS, dtype=np.float32)
        self.team_tricks = [0, 0]
        self.tricks_played = 0
        self.done = False

        return self._get_obs(), self._get_info()

    def step(self, action: int):
        if self.done:
            raise RuntimeError("Episode is done. Call reset().")

        # Validate action
        valid = self.action_mask()
        if valid[action] == 0:
            # If invalid action, pick first valid card
            action = int(np.argmax(valid))

        card = action
        player = self.current_player

        # Play card
        self.hands[player].remove(card)
        self.trick_cards.append((player, card))
        self.played_cards[card] = 1.0

        reward = 0.0
        terminated = False
        truncated = False

        if len(self.trick_cards) == NUM_PLAYERS:
            # Resolve trick
            winner = self._resolve_trick()
            winning_team = TEAMS[winner]
            self.team_tricks[winning_team] += 1
            self.tricks_played += 1

            # Reward for the acting player's team
            acting_team = TEAMS[player]
            if winning_team == acting_team:
                reward = 1.0
            else:
                reward = -1.0

            self.trick_cards = []
            self.lead_player = winner
            self.current_player = winner

            if self.tricks_played == NUM_TRICKS:
                # Round over — bonus for winning team
                if self.team_tricks[acting_team] > self.team_tricks[1 - acting_team]:
                    reward += 5.0
                else:
                    reward -= 5.0
                terminated = True
                self.done = True
        else:
            self.current_player = (self.current_player + 1) % NUM_PLAYERS

        obs = self._get_obs()
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def action_mask(self) -> np.ndarray:
        """Return a binary mask of valid actions for the current player."""
        mask = np.zeros(NUM_CARDS, dtype=np.float32)
        hand = self.hands[self.current_player]

        if not hand:
            return mask

        if self.trick_cards:
            # Must follow lead suit if possible
            lead_suit = self.trick_cards[0][1] // 13
            follow = [c for c in hand if c // 13 == lead_suit]
            if follow:
                for c in follow:
                    mask[c] = 1.0
                return mask

        # No constraint — can play any card in hand
        for c in hand:
            mask[c] = 1.0
        return mask

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        """Build observation vector for the current player."""
        obs = np.zeros(162, dtype=np.float32)

        # Own hand (0-51)
        for c in self.hands[self.current_player]:
            obs[c] = 1.0

        # Played cards (52-103)
        obs[52:104] = self.played_cards

        # Current trick cards (104-155)
        for _, card in self.trick_cards:
            obs[104 + card] = 1.0

        # Trump suit one-hot (156-159)
        obs[156 + self.trump_suit] = 1.0

        # Team tricks normalised (160-161)
        obs[160] = self.team_tricks[0] / 13.0
        obs[161] = self.team_tricks[1] / 13.0

        return obs

    def _get_info(self) -> dict:
        return {
            "current_player": self.current_player,
            "trump_suit": self.trump_suit,
            "team_tricks": list(self.team_tricks),
            "tricks_played": self.tricks_played,
            "action_mask": self.action_mask(),
        }

    def _resolve_trick(self) -> int:
        """Determine the winner of the current trick."""
        lead_suit = self.trick_cards[0][1] // 13

        best_player = self.trick_cards[0][0]
        best_card = self.trick_cards[0][1]
        best_is_trump = (best_card // 13) == self.trump_suit

        for player, card in self.trick_cards[1:]:
            card_suit = card // 13
            is_trump = card_suit == self.trump_suit

            if is_trump and not best_is_trump:
                # Trump beats non-trump
                best_player, best_card, best_is_trump = player, card, True
            elif is_trump and best_is_trump:
                # Higher trump wins
                if card % 13 > best_card % 13:
                    best_player, best_card = player, card
            elif card_suit == lead_suit and not best_is_trump:
                # Same suit as lead, higher rank wins
                if card % 13 > best_card % 13:
                    best_player, best_card = player, card
            # Off-suit non-trump cards cannot win

        return best_player

    def render(self):
        if self.render_mode != "human":
            return
        print(f"\n--- Trick {self.tricks_played + 1} ---")
        print(f"Trump: {SUITS[self.trump_suit]}")
        print(f"Team tricks: {self.team_tricks}")
        print(f"Current player: {self.current_player}")
        if self.trick_cards:
            print("Cards on table:")
            for p, c in self.trick_cards:
                print(f"  Player {p}: {card_name(c)}")
        print(f"Hand: {[card_name(c) for c in self.hands[self.current_player]]}")


class SelfPlayWrapper(gym.Wrapper):
    """Wrapper that handles self-play: a single model controls all 4 players.

    Each call to step() may internally advance multiple players (the opponents)
    using the provided policy before returning. From the RL algorithm's
    perspective, it looks like a single-agent environment.
    """

    def __init__(self, env, policy_fn=None):
        super().__init__(env)
        self.policy_fn = policy_fn  # callable(obs, mask) -> action

    def set_policy(self, policy_fn):
        """Set the policy function used for opponent moves."""
        self.policy_fn = policy_fn

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._learning_player = self.env.current_player
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        if terminated or truncated:
            return obs, reward, terminated, truncated, info

        # Let other players play until it's the learning player's turn again
        while self.env.current_player != self._learning_player:
            if self.policy_fn is not None:
                mask = self.env.action_mask()
                other_obs = self.env._get_obs()
                other_action = self.policy_fn(other_obs, mask)
            else:
                # Random policy fallback
                mask = self.env.action_mask()
                valid_actions = np.where(mask > 0)[0]
                other_action = self.np_random.choice(valid_actions)

            obs, r, terminated, truncated, info = self.env.step(other_action)
            if terminated or truncated:
                # Compute reward from learning player's perspective
                team = TEAMS[self._learning_player]
                other_team = 1 - team
                tricks_won = self.env.team_tricks[team]
                tricks_lost = self.env.team_tricks[other_team]
                reward = (tricks_won - tricks_lost) + (
                    5.0 if tricks_won > tricks_lost else -5.0
                )
                return obs, reward, terminated, truncated, info

        return obs, reward, terminated, truncated, info
