import tempfile
import unittest
from pathlib import Path

import torch

from whist_ai.engine import WhistGame
from whist_ai.model import OBSERVATION_SIZE, PolicyNetwork, encode_observation, load_checkpoint, save_checkpoint
from whist_ai.policies import RulePolicy
from whist_ai.simulator import GameController


class ModelTests(unittest.TestCase):
    def test_observation_encoding_has_stable_size(self):
        controller = GameController(WhistGame(seed=21), [RulePolicy()] * 4)
        encoded = encode_observation(controller.observation(0))
        self.assertEqual(encoded.shape, (OBSERVATION_SIZE,))
        self.assertEqual(encoded.dtype.name, "float32")

    def test_observation_contains_public_card_and_bid_history(self):
        controller = GameController(WhistGame(seed=21), [RulePolicy()] * 4)
        controller.step()
        observation = controller.observation(controller.game.current_player)
        self.assertEqual(len(observation["played_cards"]), 0)
        self.assertEqual(len(observation["bid_history"]), 1)
        self.assertEqual(encode_observation(observation).shape, (OBSERVATION_SIZE,))

    def test_checkpoint_round_trip(self):
        network = PolicyNetwork()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.pt"
            save_checkpoint(network, path)
            restored = load_checkpoint(path)
        for first, second in zip(network.parameters(), restored.parameters()):
            self.assertTrue(torch.equal(first, second))


if __name__ == "__main__":
    unittest.main()
