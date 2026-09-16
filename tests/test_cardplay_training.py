import unittest

from training.train_cardplay import CONTRACTS, SCENARIO_CYCLE, scenario_for_episode


class CardplayScheduleTests(unittest.TestCase):
    def test_cycle_covers_each_contract_and_learner_declarer_pair(self):
        scenarios = [scenario_for_episode(episode) for episode in range(1, SCENARIO_CYCLE + 1)]
        self.assertEqual(len(set(scenarios)), len(CONTRACTS) * 16)
        for contract in CONTRACTS:
            matching = [(learner, declarer) for value, learner, declarer in scenarios if value == contract]
            self.assertEqual(len(matching), 16)
            self.assertEqual(set(matching), {(learner, declarer) for learner in range(4) for declarer in range(4)})