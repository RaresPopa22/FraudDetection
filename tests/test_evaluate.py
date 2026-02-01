import numpy as np
from src.evaluate import find_optimal_threshold
from sklearn.metrics import f1_score

class TestFindOptimalThreshold:

    def test_returns_between_0_and_1(self):
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.1, 0.4, 0.5, 0.99])

        threshold = find_optimal_threshold(y_true, y_proba)
        assert 0 <= threshold <= 1

    def test_perfect_separation(self):
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.1, 0.4, 0.5, 0.99])

        # T         P         R       F1
        # 0.1       0.5       1       0.67
        # 0.4       0.67      1       0.80
        # 0.5       1         1       1
        # 0.99      1         0.5     0.67

        threshold = find_optimal_threshold(y_true, y_proba)
        assert threshold == 0.5
