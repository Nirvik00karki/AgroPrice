from django.test import TestCase
from price_prediction.views import ridge_regression_fit, ridge_regression_predict
import numpy as np

class RidgeRegressionTests(TestCase):

    def test_coefficient_calculation(self):
        # Generate synthetic data
        X = np.random.rand(100, 5)
        y = np.random.rand(100)
        alpha = 1.0

        coefficients = ridge_regression_fit(X, y, alpha)

        self.assertIsNotNone(coefficients)

    def test_prediction(self):
        X = np.random.rand(100, 5)
        coefficients = np.random.rand(6)

        predictions = ridge_regression_predict(coefficients, X)

        # Assert that predictions are calculated successfully
        self.assertIsNotNone(predictions)
