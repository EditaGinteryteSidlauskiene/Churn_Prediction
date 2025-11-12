import numpy as np

class ThresholdedModel:
    """
    Make 0.5 correspond to your custom threshold t for any classifier with predict_proba.
    """
    def __init__(self, base_model, threshold: float):
        self.base = base_model
        self.t = float(threshold)

    def predict_proba(self, X):
        p = self.base.predict_proba(X)[:, 1]
        t = self.t
        below = 0.5 * (p / t)
        above = 0.5 + 0.5 * ((p - t) / (1 - t))
        s = np.where(p < t, below, above).clip(0, 1)
        return np.c_[1.0 - s, s]

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)