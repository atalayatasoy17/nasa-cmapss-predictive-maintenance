import unittest

import pandas as pd

from src.alerts import add_confirmed_alerts


class AlertRuleTests(unittest.TestCase):
    def test_three_consecutive_predictions_and_reset(self):
        history = pd.DataFrame({
            "unit_id": [1] * 8,
            "cycle": list(range(1, 9)),
            "predicted_RUL_capped": [40, 29, 28, 35, 29, 28, 27, 40],
        })

        result = add_confirmed_alerts(history)

        self.assertEqual(
            result["confirmed_now"].tolist(),
            [False, False, False, False, False, False, True, False],
        )
        self.assertEqual(
            result["ever_confirmed"].tolist(),
            [False, False, False, False, False, False, True, True],
        )

    def test_engines_are_checked_separately(self):
        history = pd.DataFrame({
            "unit_id": [1, 1, 2, 2, 2],
            "cycle": [1, 2, 1, 2, 3],
            "predicted_RUL_capped": [20, 20, 20, 20, 20],
        })

        result = add_confirmed_alerts(history)

        self.assertEqual(
            result["confirmed_now"].tolist(),
            [False, False, False, False, True],
        )

    def test_rejects_missing_cycle(self):
        history = pd.DataFrame({
            "unit_id": [1, 1, 1],
            "cycle": [1, 2, 4],
            "predicted_RUL_capped": [20, 20, 20],
        })

        with self.assertRaisesRegex(ValueError, "Cycles must be consecutive"):
            add_confirmed_alerts(history)


if __name__ == "__main__":
    unittest.main()