import unittest

from esmakker_env import NO_TRUMP, PASKRIG, EsmakkerGame
from esmakker_rl_env import (
    OPENING_PASS_PENALTY,
    OPPONENT_COUNTER_BID_PROBABILITY,
    PASS_ACTION,
    EsmakkerEnv,
)


class PaskrigTests(unittest.TestCase):
    def test_all_pass_starts_paskrig(self):
        game = EsmakkerGame(seed=1)

        for _ in range(4):
            game.pass_bid(game.current_player)

        self.assertEqual(game.current_bid, PASKRIG)
        self.assertEqual(game.phase, "play")
        self.assertIsNone(game.declarer)
        self.assertEqual(game.trump_suit, NO_TRUMP)

    def test_ace_is_low_when_resolving_paskrig_trick(self):
        game = EsmakkerGame(seed=1)
        game.current_bid = PASKRIG

        winner = game.trick_winner([
            (0, 12),  # Ace of clubs
            (1, 0),   # Two of clubs
            (2, 11),  # King of clubs
            (3, 10),  # Queen of clubs
        ])

        self.assertEqual(winner, 2)

    def test_ace_is_low_for_sol(self):
        game = EsmakkerGame(seed=1)
        game.current_bid = "sol"

        winner = game.trick_winner([
            (0, 12),  # Ace of clubs
            (1, 0),   # Two of clubs
            (2, 11),  # King of clubs
            (3, 10),  # Queen of clubs
        ])

        self.assertEqual(winner, 2)

    def test_single_winner_payout_is_zero_sum(self):
        game = EsmakkerGame(seed=1)
        game.current_bid = PASKRIG
        game.tricks_won = [1, 4, 4, 4]

        settlement = game.settle()

        self.assertEqual(settlement.payments, (9.0, -3.0, -3.0, -3.0))
        self.assertAlmostEqual(sum(settlement.payments), 0.0)

    def test_tied_winners_split_combined_pot(self):
        game = EsmakkerGame(seed=1)
        game.current_bid = PASKRIG
        game.tricks_won = [1, 1, 5, 6]

        settlement = game.settle()

        self.assertEqual(settlement.payments, (4.5, 4.5, -4.0, -5.0))
        self.assertAlmostEqual(sum(settlement.payments), 0.0)

    def test_opening_pass_receives_immediate_penalty(self):
        env = EsmakkerEnv()
        env.game = EsmakkerGame(seed=1)
        env.learning_player = env.game.current_player
        env.done = False

        _, reward, terminated, _, info = env.step(PASS_ACTION)

        self.assertFalse(terminated)
        self.assertEqual(info["opening_pass_penalty"], OPENING_PASS_PENALTY)
        self.assertAlmostEqual(reward, -OPENING_PASS_PENALTY)

    def test_later_pass_is_reward_neutral(self):
        env = EsmakkerEnv()
        env.game = EsmakkerGame(seed=1)
        env.game.current_bid = "7"
        env.game.declarer = (env.game.current_player - 1) % 4
        env.learning_player = env.game.current_player
        env.done = False

        _, reward, terminated, _, info = env.step(PASS_ACTION)

        self.assertFalse(terminated)
        self.assertNotIn("pass_penalty", info)
        self.assertAlmostEqual(reward, 0.0)

    def test_non_declarer_cannot_profit_from_failed_opponent_contract(self):
        env = EsmakkerEnv()
        env.game = EsmakkerGame(seed=1)
        env.learning_player = 0
        env.game.current_bid = "bordlaegger"
        env.game.declarer = 1
        env.game.tricks_won[1] = 2
        env.game.phase = "complete"
        env.game.round_settlement = env.game.settle()
        self.assertEqual(env.game.round_settlement.payments[0], 8)
        self.assertAlmostEqual(env._settlement_reward(), 0.0)

    def test_contract_13_uses_reduced_base_value(self):
        game = EsmakkerGame(seed=1)
        game.current_bid = "13"
        game.declarer = 0
        game.partner_player = 1
        game.tricks_won = [7, 6, 0, 0]

        settlement = game.settle()

        self.assertTrue(settlement.contract_succeeded)
        self.assertEqual(settlement.value, 25)
        self.assertEqual(settlement.payments, (25, 25, -25, -25))

    def test_successful_contract_13_keeps_full_normalized_reward(self):
        env = EsmakkerEnv()
        env.game = EsmakkerGame(seed=1)
        env.game.current_bid = "13"
        env.game.declarer = 0
        env.game.partner_player = 1
        env.game.tricks_won = [7, 6, 0, 0]
        env.game.phase = "complete"
        env.learning_player = 0
        env.game.round_settlement = env.game.settle()

        self.assertEqual(env._settlement_reward(), 1.0)

    def test_opponent_counter_bid_probability_is_configured(self):
        self.assertEqual(OPPONENT_COUNTER_BID_PROBABILITY, 0.15)


if __name__ == "__main__":
    unittest.main()