import unittest

from training.cardplay.esmakker_cardplay_env import EsmakkerCardPlayEnv


class EsmakkerCardPlayTests(unittest.TestCase):
    def test_reset_exposes_only_legal_card_actions(self):
        env = EsmakkerCardPlayEnv(contract="7")
        observation, info = env.reset(seed=7)

        self.assertEqual(observation.shape, env.observation_space.shape)
        self.assertEqual(info["contract"], "7")
        self.assertGreater(env.action_masks().sum(), 0)
        self.assertLessEqual(env.action_masks().sum(), 13)

    def test_fixed_contract_round_completes(self):
        env = EsmakkerCardPlayEnv(contract="7")
        _, _ = env.reset(seed=11)
        terminated = False
        steps = 0
        while not terminated:
            action = int(next(card for card, legal in enumerate(env.action_masks()) if legal))
            _, _, terminated, _, info = env.step(action)
            steps += 1

        self.assertTrue(terminated)
        self.assertEqual(steps, 13)
        self.assertIsNotNone(info["settlement"])
        self.assertEqual(sum(info["tricks_won"]), 13)

    def test_team_tricks_does_not_double_count_declarer_partner(self):
        env = EsmakkerCardPlayEnv(contract="7")
        matching_seed = next(
            seed
            for seed in range(100)
            if (env.reset(seed=seed)[1]["declarer"] == env.game.partner_player)
        )
        env.reset(seed=matching_seed)
        env.game.tricks_won = [1, 2, 3, 4]

        expected = env.game.tricks_won[env.learning_player]
        self.assertEqual(env._team_tricks(), expected)
        self.assertLessEqual(env._team_tricks(), 13)


if __name__ == "__main__":
    unittest.main()
