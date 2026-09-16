import unittest

from training.train_declarer_cardplay import DEFAULT_CONTRACTS, parse_contracts


class DeclarerCardplayTrainingTests(unittest.TestCase):
    def test_default_curriculum_starts_with_numeric_foundation(self):
        self.assertEqual(DEFAULT_CONTRACTS, (7, 8, 9))

    def test_contracts_are_parsed_in_order(self):
        self.assertEqual(parse_contracts("7, 10,13"), (7, 10, 13))

    def test_contracts_reject_non_numeric_or_out_of_range_values(self):
        with self.assertRaises(ValueError):
            parse_contracts("sol,8")
        with self.assertRaises(ValueError):
            parse_contracts("6,8")
        with self.assertRaises(ValueError):
            parse_contracts("")


if __name__ == "__main__":
    unittest.main()
