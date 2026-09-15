import unittest

from whist_ai.rewards import nolo_reward, numeric_reward, paskrig_rewards


class NumericRewardTests(unittest.TestCase):
    def test_success_counts_from_seven_and_includes_bid_value(self):
        self.assertEqual(numeric_reward(7, 7), 1)
        self.assertEqual(numeric_reward(8, 8), 3)
        self.assertEqual(numeric_reward(13, 13), 13)

    def test_failed_numeric_bid_charges_two_per_missed_trick(self):
        self.assertEqual(numeric_reward(8, 7), -3)
        self.assertEqual(numeric_reward(8, 6), -5)
        self.assertEqual(numeric_reward(12, 10), -9)

    def test_failed_13_has_locked_in_surcharge(self):
        self.assertEqual(numeric_reward(13, 12), -14)
        self.assertEqual(numeric_reward(13, 10), -18)


class NoloRewardTests(unittest.TestCase):
    def test_nolo_success_and_failure_values(self):
        self.assertEqual(nolo_reward("Sol", 1), 4)
        self.assertEqual(nolo_reward("Sol", 2), -6)
        self.assertEqual(nolo_reward("Ren sol", 0), 8)
        self.assertEqual(nolo_reward("Ren sol", 1), -12)
        self.assertEqual(nolo_reward("Bordlaegger", 0), 12)
        self.assertEqual(nolo_reward("Bordlaegger", 1), -18)


class PaskrigRewardTests(unittest.TestCase):
    def test_relative_rewards_with_one_winner(self):
        self.assertEqual(paskrig_rewards([1, 2, 4, 6]), (5, -2, -6, -10))

    def test_relative_rewards_with_tied_winners(self):
        self.assertEqual(paskrig_rewards([1, 1, 2, 9]), (8, 8, -2, -16))


if __name__ == "__main__":
    unittest.main()
