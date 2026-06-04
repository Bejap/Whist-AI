"""Whist card game environment compatible with Gymnasium."""

import gymnasium as gym
import numpy as np
from gymnasium import spaces


# Card encoding: 0-51, suit = card // 13, rank = card % 13
SUITS = ["Clubs", "Diamonds", "Hearts", "Spades"]
NO_TRUMP = 4  # sentinel value: no trump suit this round
NO_LEAD_SUIT = 4  # sentinel value in lead-suit one-hot when trick is empty
RANKS = [
    "2", "3", "4", "5", "6", "7", "8", "9", "10",
    "Jack", "Queen", "King", "Ace",
]
NUM_CARDS = 52
NUM_PLAYERS = 4
CARDS_PER_PLAYER = 13
NUM_TRICKS = 13

# Reward values
TRICK_WIN_REWARD = 2.0
TRICK_LOSS_REWARD = -2.0
TERMINAL_WIN_REWARD = 2.0
TERMINAL_LOSS_REWARD = -2.0

# Observation layout sizes
OBS_HAND = NUM_CARDS
OBS_PLAYED_BY_PLAYER = NUM_PLAYERS * NUM_CARDS
OBS_CURRENT_TRICK = NUM_CARDS
OBS_TRUMP = 5
OBS_TEAM_TRICKS = 2
OBS_PLAYER_ID = NUM_PLAYERS
OBS_TRICK_WINNER = NUM_PLAYERS
OBS_LEAD_SUIT = 5
OBS_TRUMP_VOID = NUM_PLAYERS
OBS_TRICK_POSITION = NUM_PLAYERS
OBS_SIZE = (
    OBS_HAND
    + OBS_PLAYED_BY_PLAYER
    + OBS_CURRENT_TRICK
    + OBS_TRUMP
    + OBS_TEAM_TRICKS
    + OBS_PLAYER_ID
    + OBS_TRICK_WINNER
    + OBS_LEAD_SUIT
    + OBS_TRUMP_VOID
    + OBS_TRICK_POSITION
)

# Teams: team 0 = players 0, 2; team 1 = players 1, 3
TEAMS = {0: 0, 1: 1, 2: 0, 3: 1}


def card_name(card_id: int) -> str:
    """Return a human-readable card name."""
    return f"{RANKS[card_id % 13]} of {SUITS[card_id // 13]}"


def trump_name(trump_suit: int) -> str:
    """Return a human-readable trump name."""
    if trump_suit == NO_TRUMP:
        return "No Trump"
    return SUITS[trump_suit]


class WhistEnv(gym.Env):
    """Gymnasium environment for 4-player Whist.

    Observation (length 340):
        - own hand:              52 bits (one-hot)
        - played cards by seat: 208 bits (4 × 52)
        - current trick:         52 bits (cards on the table this trick, up to 3)
        - trump suit:             5 bits (one-hot; index 0-3 = suit, index 4 = no trump)
        - team tricks:            2 floats (team0 tricks / 13, team1 tricks / 13)
        - learning player id:     4 bits (one-hot encoding of seat 0-3)
        - current trick winner:   4 bits (one-hot of player currently winning trick)
        - lead suit:              5 bits (one-hot; index 4 = no lead yet)
        - trump exhaustion:       4 bits (one flag per seat)
        - trick position:         4 bits (one-hot position in trick for observing seat)
    Total = 52 + 208 + 52 + 5 + 2 + 4 + 4 + 5 + 4 + 4 = 340

    Action space: Discrete(52), masked to valid cards in hand.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(OBS_SIZE,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(NUM_CARDS)

        # State variables
        self.hands = [[] for _ in range(NUM_PLAYERS)]
        self.trump_suit = 0
        self.current_player = 0
        self.trick_cards = []  # list of (player, card) for current trick
        self.lead_player = 0
        self.played_cards_by_player = np.zeros(
            (NUM_PLAYERS, NUM_CARDS), dtype=np.float32
        )
        self.trump_void = np.zeros(NUM_PLAYERS, dtype=np.float32)
        self.team_tricks = [0, 0]
        self.tricks_played = 0
        self.done = False
        self.last_trick_winning_team = None
        self.last_trick_capture_bonus = 0.0

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
        self.trump_suit = self.np_random.integers(0, 5)  # 0-3 = suit, 4 = no trump
        self.current_player = 0
        self.lead_player = 0
        self.trick_cards = []
        self.played_cards_by_player = np.zeros(
            (NUM_PLAYERS, NUM_CARDS), dtype=np.float32
        )
        self.trump_void = np.zeros(NUM_PLAYERS, dtype=np.float32)
        self.team_tricks = [0, 0]
        self.tricks_played = 0
        self.done = False
        self.last_trick_winning_team = None
        self.last_trick_capture_bonus = 0.0

        return self._get_obs(), self._get_info()

    def step(self, action: int):
        if self.done:
            raise RuntimeError("Episode is done. Call reset().")

        # Validate action — penalise invalid picks
        valid = self.action_mask()
        invalid_penalty = 0.0
        if valid[action] == 0:
            invalid_penalty = -0.5
            action = int(np.argmax(valid))

        card = action
        player = self.current_player

        # Infer trump exhaustion: if trump is led and a player cannot follow trump.
        if self.trick_cards and self.trump_suit != NO_TRUMP:
            lead_suit = self.trick_cards[0][1] // 13
            if lead_suit == self.trump_suit and (card // 13) != self.trump_suit:
                self.trump_void[player] = 1.0

        # Play card
        self.hands[player].remove(card)
        self.trick_cards.append((player, card))
        self.played_cards_by_player[player, card] = 1.0

        reward = invalid_penalty
        terminated = False
        truncated = False
        self.last_trick_winning_team = None
        self.last_trick_capture_bonus = 0.0

        if len(self.trick_cards) == NUM_PLAYERS:
            # Resolve trick
            winner = self._resolve_trick()
            winning_team = TEAMS[winner]
            self.team_tricks[winning_team] += 1
            self.tricks_played += 1

            # Reward for the acting player's team
            acting_team = TEAMS[player]
            if winning_team == acting_team:
                reward += TRICK_WIN_REWARD
            else:
                reward += TRICK_LOSS_REWARD

            # Bonus for capturing high-value opponent cards
            capture_bonus = self._capture_bonus_for_team(winning_team)
            if winning_team == acting_team:
                reward += capture_bonus
            self.last_trick_winning_team = winning_team
            self.last_trick_capture_bonus = capture_bonus

            # --- Reward shaping ---
            reward += self._shape_reward(player, card, winner)

            self.trick_cards = []
            self.lead_player = winner
            self.current_player = winner

            if self.tricks_played == NUM_TRICKS:
                # Round over — terminal bonus
                if self.team_tricks[acting_team] > self.team_tricks[1 - acting_team]:
                    reward += TERMINAL_WIN_REWARD
                else:
                    reward += TERMINAL_LOSS_REWARD
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

    def _get_obs(self, player_id=None) -> np.ndarray:
        """Build observation vector for the current player.

        Args:
            player_id: Optional seat id (0-3) to encode in the observation.
                       When None, defaults to current_player.
        """
        obs = np.zeros(OBS_SIZE, dtype=np.float32)
        idx = 0

        # Own hand
        pid = player_id if player_id is not None else self.current_player
        for c in self.hands[pid]:
            obs[idx + c] = 1.0
        idx += OBS_HAND

        # Played cards by player
        obs[idx : idx + OBS_PLAYED_BY_PLAYER] = self.played_cards_by_player.reshape(-1)
        idx += OBS_PLAYED_BY_PLAYER

        # Current trick cards
        for _, card in self.trick_cards:
            obs[idx + card] = 1.0
        idx += OBS_CURRENT_TRICK

        # Trump suit one-hot
        obs[idx + self.trump_suit] = 1.0
        idx += OBS_TRUMP

        # Team tricks normalised
        obs[idx] = self.team_tricks[0] / 13.0
        obs[idx + 1] = self.team_tricks[1] / 13.0
        idx += OBS_TEAM_TRICKS

        # Learning player id one-hot
        pid = player_id if player_id is not None else self.current_player
        obs[idx + pid] = 1.0
        idx += OBS_PLAYER_ID

        # Current trick winner one-hot
        if self.trick_cards:
            tw = self._current_trick_winner()
            obs[idx + tw] = 1.0
        idx += OBS_TRICK_WINNER

        # Lead suit one-hot (index 4 means no lead yet)
        if self.trick_cards:
            lead_suit = self.trick_cards[0][1] // 13
            obs[idx + lead_suit] = 1.0
        else:
            obs[idx + NO_LEAD_SUIT] = 1.0
        idx += OBS_LEAD_SUIT

        # Trump exhaustion per seat
        obs[idx : idx + OBS_TRUMP_VOID] = self.trump_void
        idx += OBS_TRUMP_VOID

        # Trick position (seat index relative to lead seat)
        lead = self.trick_cards[0][0] if self.trick_cards else self.current_player
        trick_pos = (pid - lead) % NUM_PLAYERS
        obs[idx + trick_pos] = 1.0

        return obs

    def _get_info(self) -> dict:
        return {
            "current_player": self.current_player,
            "trump_suit": self.trump_suit,
            "team_tricks": list(self.team_tricks),
            "tricks_played": self.tricks_played,
            "trump_void": self.trump_void.astype(int).tolist(),
            "action_mask": self.action_mask(),
            "last_trick_winning_team": self.last_trick_winning_team,
            "last_trick_capture_bonus": self.last_trick_capture_bonus,
        }

    def _capture_bonus_for_team(self, team: int) -> float:
        """Bonus for capturing high-value opponent cards in a won trick."""
        bonus = 0.0
        for player, card in self.trick_cards:
            if TEAMS[player] == team:
                continue
            rank = card % 13
            if rank == 12:      # Ace
                bonus += 0.3
            elif rank == 11:    # King
                bonus += 0.3
            elif rank == 10:    # Queen
                bonus += 0.2
        return bonus

    def _shape_reward(self, player: int, card: int, winner: int) -> float:
        """Compute bonus / penalty shaping for the trick just resolved.

        Must be called *before* trick_cards is cleared.
        """
        bonus = 0.0
        card_suit = card // 13
        card_rank = card % 13
        has_trump = self.trump_suit != NO_TRUMP
        is_trump = has_trump and card_suit == self.trump_suit
        acting_team = TEAMS[player]
        winning_team = TEAMS[winner]
        lead_suit = self.trick_cards[0][1] // 13

        # Determine who was winning *before* the acting player played
        # by looking at all trick cards except the acting player's.
        cards_before = [(p, c) for p, c in self.trick_cards if p != player]
        winner_before = None
        if cards_before:
            winner_before = self._peek_trick_winner(cards_before)

        if winning_team == acting_team:
            if is_trump and winner == player:
                # Efficient trump bonus: won trick with trump (+0.4)
                bonus += 0.4

                # Smart trump bonus: played the lowest winning trump (+0.5)
                trump_cards_in_hand = [
                    c for c in self.hands[player]
                    if c // 13 == self.trump_suit
                ]
                # Include the card just played (already removed from hand)
                all_trumps = sorted(
                    trump_cards_in_hand + [card], key=lambda c: c % 13
                )
                # Find the lowest trump that would have won
                # Need to beat all other cards in the trick
                best_opponent_rank = -1
                best_opponent_is_trump = False
                for p, c in self.trick_cards:
                    if p == player:
                        continue
                    c_suit = c // 13
                    c_is_trump = has_trump and c_suit == self.trump_suit
                    if c_is_trump:
                        if c % 13 > best_opponent_rank or not best_opponent_is_trump:
                            best_opponent_rank = c % 13
                            best_opponent_is_trump = True
                    elif c_suit == lead_suit and not best_opponent_is_trump:
                        if c % 13 > best_opponent_rank:
                            best_opponent_rank = c % 13

                # Find lowest trump that beats the best opponent card
                lowest_winning_trump = None
                for t in all_trumps:
                    t_rank = t % 13
                    if best_opponent_is_trump:
                        if t_rank > best_opponent_rank:
                            lowest_winning_trump = t
                            break
                    else:
                        # Any trump beats non-trump
                        lowest_winning_trump = t
                        break

                if lowest_winning_trump is not None and card == lowest_winning_trump:
                    bonus += 0.5

            elif card_suit == lead_suit and winner == player:
                # Won with highest card of lead suit
                bonus += 0.2
        else:
            # Team lost the trick
            if is_trump:
                # Wasted a trump on a trick the team lost
                bonus -= 0.1

            # Must-trump penalty: had no lead suit, had trump, didn't play trump
            if not is_trump and has_trump:
                player_has_lead = any(
                    c // 13 == lead_suit for c in self.hands[player]
                )
                player_has_trump = any(
                    c // 13 == self.trump_suit for c in self.hands[player]
                )
                # Player couldn't follow suit (otherwise they would have been
                # forced to), so check if they had trump available
                if not player_has_lead and player_has_trump:
                    bonus -= 0.4

        # Penalise wasting trump when teammate already winning
        if is_trump and winning_team == acting_team and winner != player:
            bonus -= 0.1

        # Wasted high card penalty: teammate was already winning and player
        # threw a high card (rank >= Jack, i.e. rank index >= 9)
        if (winner_before is not None
                and TEAMS[winner_before] == acting_team
                and winner_before != player
                and card_rank >= 9):
            bonus -= 0.3

        return bonus

    def _current_trick_winner(self) -> int:
        """Return the player currently winning the trick (without resolving).

        Assumes trick_cards is non-empty.
        """
        return self._peek_trick_winner(self.trick_cards)

    def _peek_trick_winner(self, cards) -> int:
        """Determine the winner among a list of (player, card) entries."""
        lead_suit = cards[0][1] // 13
        has_trump = self.trump_suit != NO_TRUMP

        best_player = cards[0][0]
        best_card = cards[0][1]
        best_is_trump = has_trump and (best_card // 13) == self.trump_suit

        for p, c in cards[1:]:
            c_suit = c // 13
            c_is_trump = has_trump and c_suit == self.trump_suit

            if c_is_trump and not best_is_trump:
                best_player, best_card, best_is_trump = p, c, True
            elif c_is_trump and best_is_trump:
                if c % 13 > best_card % 13:
                    best_player, best_card = p, c
            elif c_suit == lead_suit and not best_is_trump:
                if c % 13 > best_card % 13:
                    best_player, best_card = p, c

        return best_player

    def _resolve_trick(self) -> int:
        """Determine the winner of the current trick."""
        return self._peek_trick_winner(self.trick_cards)

    def render(self):
        if self.render_mode != "human":
            return
        print(f"\n--- Trick {self.tricks_played + 1} ---")
        print(f"Trump: {trump_name(self.trump_suit)}")
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

    def __init__(self, env, policy_fn=None, epsilon=0.0):
        super().__init__(env)
        self.policy_fn = policy_fn  # callable(obs, mask) -> action
        self.epsilon = epsilon      # probability of random opponent action

    def set_policy(self, policy_fn):
        """Set the policy function used for opponent moves."""
        self.policy_fn = policy_fn

    def set_epsilon(self, epsilon: float):
        """Set the epsilon for opponent randomization."""
        self.epsilon = epsilon

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._learning_player = self.env.current_player
        # Rebuild obs with learning player id
        obs = self.env._get_obs(player_id=self._learning_player)
        info = self.env._get_info()
        return obs, info

    def step(self, action):
        team = TEAMS[self._learning_player]

        # Record trick counts before the learning player's action
        tricks_before = self.env.team_tricks[team]

        obs, reward, terminated, truncated, info = self.env.step(action)

        if terminated or truncated:
            obs = self.env._get_obs(player_id=self._learning_player)
            return obs, reward, terminated, truncated, info

        # Let other players play until it's the learning player's turn again
        while self.env.current_player != self._learning_player:
            mask = self.env.action_mask()
            valid_actions = np.where(mask > 0)[0]

            if self.policy_fn is not None:
                # Epsilon-greedy: random valid action with probability epsilon
                if self.epsilon > 0 and self.env.np_random.random() < self.epsilon:
                    other_action = int(self.env.np_random.choice(valid_actions))
                else:
                    other_obs = self.env._get_obs()
                    other_action = self.policy_fn(other_obs, mask)
            else:
                # Random policy fallback
                other_action = int(self.env.np_random.choice(valid_actions))

            tricks_before_step = list(self.env.team_tricks)
            obs, _r, terminated, truncated, info = self.env.step(other_action)

            # If a trick resolved during an opponent turn, credit the
            # learning player with +2 (team won) or -2 (team lost).
            if self.env.team_tricks[0] != tricks_before_step[0] or self.env.team_tricks[1] != tricks_before_step[1]:
                if self.env.team_tricks[team] > tricks_before_step[team]:
                    reward += TRICK_WIN_REWARD
                    if info.get("last_trick_winning_team") == team:
                        reward += float(info.get("last_trick_capture_bonus", 0.0))
                else:
                    reward += TRICK_LOSS_REWARD

            if terminated or truncated:
                obs = self.env._get_obs(player_id=self._learning_player)
                return obs, reward, terminated, truncated, info

        # Rebuild obs with learning player id
        obs = self.env._get_obs(player_id=self._learning_player)
        info = self.env._get_info()
        return obs, reward, terminated, truncated, info
