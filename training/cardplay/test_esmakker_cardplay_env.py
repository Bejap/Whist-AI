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

    def test_all_contract_mode_samples_numeric_contracts(self):
        env = EsmakkerCardPlayEnv(contract="all")
        contracts = {env.reset(seed=seed)[1]["contract"] for seed in range(30)}

        self.assertGreater(len(contracts), 1)
        self.assertTrue(contracts <= {str(number) for number in range(7, 14)})

    def test_observation_tracks_played_cards_and_void_suits(self):
        env = EsmakkerCardPlayEnv(contract="7")
        env.reset(seed=11)
        action = int(next(card for card, legal in enumerate(env.action_masks()) if legal))
        env.step(action)

        observation = env._observation()
        played_cards = observation[146:198]
        self.assertGreaterEqual(sum(played_cards), 1)
        self.assertEqual(played_cards[action], 1.0)

        ownership_start = 198
        ownership = observation[ownership_start:ownership_start + 208].reshape(4, 52)
        played_player = next(player for player in range(4) if ownership[player, action])
        self.assertTrue(env.played_by[played_player, action])

        player = (env.learning_player + 1) % 4
        env.game.current_player = player
        env.game.trick_cards = [(env.learning_player, 0)]
        env.game.hands[player] = [13]
        env._play_card(player, 13)
        void_features = env._observation()[406:422]
        self.assertEqual(void_features[player * 4], 1.0)

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
