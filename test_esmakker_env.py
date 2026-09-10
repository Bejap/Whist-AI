import unittest

from esmakker_env import NO_TRUMP, PASKRIG, EsmakkerGame
from esmakker_rl_env import OPENING_PASS_PENALTY, PASS_ACTION, EsmakkerEnv


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


if __name__ == "__main__":
    unittest.main()