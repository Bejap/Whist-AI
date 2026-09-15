import unittest

from whist_ai.engine import WhistGame
from whist_ai.policies import RandomPolicy, RulePolicy
from whist_ai.simulator import GameController


class SimulatorTests(unittest.TestCase):
    def test_rule_policy_can_raise_and_styles_are_validated(self):
        observation = {
            "hand": tuple((0, rank) for rank in range(13)),
            "current_bid": 7,
        }
        legal_bids = ("pass", 8, "sol", 9, 10, "ren sol", 11, 12, "bordlaegger", 13)
        self.assertGreater(RulePolicy("aggressive").choose_bid(observation, legal_bids), 7)
        with self.assertRaises(ValueError):
            RulePolicy("reckless")

    def test_controller_runs_a_complete_game(self):
        policies = [RulePolicy() for _ in range(4)]
        controller = GameController(WhistGame(seed=8), policies)
        rewards = controller.run()
        self.assertEqual(len(rewards), 4)
        self.assertEqual(controller.game.phase, "complete")
        self.assertLessEqual(len(controller.decisions), 300)

    def test_random_controller_runs_reproducibly(self):
        def run(seed):
            policies = [RandomPolicy(seed + player) for player in range(4)]
            controller = GameController(WhistGame(seed=seed), policies)
            return controller.run(), controller.decisions

        first_rewards, first_decisions = run(12)
        second_rewards, second_decisions = run(12)
        self.assertEqual(first_rewards, second_rewards)
        self.assertEqual(first_decisions, second_decisions)

    def test_observation_does_not_reveal_partner(self):
        policies = [RulePolicy() for _ in range(4)]
        controller = GameController(WhistGame(seed=2), policies)
        observation = controller.observation(0)
        self.assertIsNone(observation["partner_player"])
        self.assertFalse(observation["partner_revealed"])

    def test_observation_reveals_partner_after_declaration(self):
        controller = GameController(WhistGame(seed=31), [RulePolicy()] * 4)
        while controller.game.phase != "choose_partner":
            controller.step()
        controller.step()
        observation = controller.observation(controller.game.current_player)
        self.assertTrue(observation["partner_revealed"])
        self.assertIsNotNone(observation["partner_player"])
        self.assertIsNotNone(observation["partner_suit"])

    def test_observation_keeps_completed_tricks_public(self):
        controller = GameController(WhistGame(seed=32), [RulePolicy()] * 4)
        while not controller.game.trick_history:
            controller.step()
        observation = controller.observation(controller.game.current_player)
        played_cards = [card for trick in controller.game.trick_history for _, card in trick]
        self.assertEqual(len(played_cards), 4)
        self.assertTrue(all(card in observation["played_cards"] for card in played_cards))


if __name__ == "__main__":
    unittest.main()
