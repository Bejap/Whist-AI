import unittest

from training.cardplay.evaluate_cardplay import summarize


class CardPlayBenchmarkTests(unittest.TestCase):
    def test_success_rate_uses_positive_learner_payment(self):
        rows = [
            {
                "contract": "9",
                "completed": 1,
                "success": 1,
                "payment": 3,
                "team_tricks": 8,
            },
            {
                "contract": "9",
                "completed": 1,
                "success": 0,
                "payment": -3,
                "team_tricks": 7,
            },
        ]

        result = summarize(rows)[0]

        self.assertEqual(result["success_rate"], 0.5)
        self.assertEqual(result["average_payment"], 0.0)

    def test_summary_separates_opponent_tables(self):
        rows = [
            {"table": "random", "contract": "7", "completed": 1, "success": 1, "payment": 1, "team_tricks": 7},
            {"table": "rules", "contract": "7", "completed": 1, "success": 0, "payment": -1, "team_tricks": 6},
        ]

        results = summarize(rows)

        self.assertEqual({result["table"] for result in results}, {"random", "rules"})


if __name__ == "__main__":
    unittest.main()