import unittest

from whist_ai.engine import WhistGame
from whist_ai.policies import RandomPolicy, RulePolicy
from whist_ai.simulator import GameController


class SimulatorTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
