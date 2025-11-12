import numpy as np
import xgboost as xgb

class XGBoosterAdapter:
    """Make an xgboost.Booster quack like an sklearn classifier."""
    def __init__(self, booster, *, objective="binary:logistic", feature_names=None):
        self.booster = booster
        self.objective = str(objective)
        self.feature_names = list(feature_names) if feature_names is not None else None

    def _proba(self, X):
        # Build DMatrix with feature names to avoid mismatch errors.
        if hasattr(X, "values") and hasattr(X, "columns"):
            dmat = xgb.DMatrix(X.values, feature_names=X.columns.tolist())
        else:
            dmat = xgb.DMatrix(X, feature_names=self.feature_names)

        p = self.booster.predict(dmat)  # shape (n,)
        # If model was trained with logit output, squash to (0,1)
        if "logitraw" in self.objective:
            p = 1.0 / (1.0 + np.exp(-p))
        return np.asarray(p, dtype=float).reshape(-1)

    def predict_proba(self, X):
        p1 = self._proba(X)
        p0 = 1.0 - p1
        return np.c_[p0, p1]

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
