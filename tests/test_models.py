"""
Unit tests for Machine Learning Models (DemandForecastModel, CancellationModel, evaluate).
"""

import unittest
from pathlib import Path
import pandas as pd
import numpy as np

from src.models.demand_forecast import DemandForecastModel
from src.models.cancellation_model import CancellationModel, generate_cancellation_training_data
from src.models.evaluate import evaluate_regression, evaluate_classification


class TestModels(unittest.TestCase):

    def test_demand_forecast_model_fit_predict(self):
        # Synthetic regression data
        np.random.seed(42)
        X = pd.DataFrame({
            "feature_1": np.random.rand(100),
            "feature_2": np.random.rand(100)
        })
        y = X["feature_1"] * 10.0 + X["feature_2"] * 5.0 + 3.0

        model = DemandForecastModel({"model_type": "ridge"})
        model.fit(X, y)
        self.assertTrue(model.is_fitted)

        preds = model.predict(X)
        self.assertEqual(len(preds), 100)
        metrics = model.evaluate(X, y, split_name="Train")
        self.assertGreater(metrics["r2"], 0.90)

    def test_cancellation_model_fit_predict_proba(self):
        X, y = generate_cancellation_training_data(n_samples=500, seed=42)
        model = CancellationModel({"model_type": "logistic"})
        model.fit(X, y)
        self.assertTrue(model.is_fitted)

        probs = model.predict_proba(X)
        self.assertEqual(len(probs), 500)
        self.assertTrue(all(0.0 <= p <= 1.0 for p in probs))

        metrics = model.evaluate(X, y, split_name="Train")
        self.assertGreater(metrics["roc_auc"], 0.70)

    def test_evaluate_regression_helper(self):
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([11.0, 19.0, 31.0])
        res = evaluate_regression(y_true, y_pred, model_name="TestReg")
        self.assertEqual(res["model"], "TestReg")
        self.assertAlmostEqual(res["mae"], 1.0, places=2)

    def test_evaluate_classification_helper(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred_proba = np.array([0.1, 0.2, 0.8, 0.9])
        res = evaluate_classification(y_true, y_pred_proba, model_name="TestClf")
        self.assertEqual(res["model"], "TestClf")
        self.assertEqual(res["accuracy"], 1.0)
        self.assertEqual(res["roc_auc"], 1.0)


if __name__ == "__main__":
    unittest.main()
