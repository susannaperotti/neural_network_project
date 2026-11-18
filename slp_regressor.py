"""Single-hidden-layer regressor with a linear output."""

import numpy as np
from sklearn.base import RegressorMixin
from sklearn.utils.validation import validate_data

from slp_base import BaseSLPEstimator


class SimpleSLPRegressor(RegressorMixin, BaseSLPEstimator):
    """Fit continuous targets with mean squared error. Score returns R².

    Losses are measured on targets standardized using optimization rows.
    Predictions are converted back to the original target units.
    """

    def _output_activation(self, logits):
        return logits

    def _output_error(self, y, y_pred):
        # MSE averages across samples and outputs.
        return 2 * (y_pred - y) / y.size

    def _compute_loss(self, y_true, y_pred):
        return float(np.mean((y_pred - y_true) ** 2))

    def fit(self, X, y):
        """Fit to a vector of targets or a matrix of multiple outputs."""
        self._validate_parameters()
        X, y = validate_data(self, X, y, dtype=float, multi_output=True, y_numeric=True)
        y = np.asarray(y, dtype=float)
        if y.size == 0 or not np.isfinite(y).all():
            raise ValueError("y must contain at least one finite numeric target per sample")
        rng = np.random.default_rng(self.random_state)
        indices = self._split_indices(y, rng)
        self._single_target = y.ndim == 1
        targets = y.reshape(len(y), -1)
        self.n_outputs_ = targets.shape[1]
        training_targets = targets[indices[0]]
        self.target_mean_ = training_targets.mean(axis=0)
        scale = training_targets.std(axis=0)
        self.target_scale_ = np.where(scale == 0, 1.0, scale)
        scaled_targets = (targets - self.target_mean_) / self.target_scale_
        self._train(X, scaled_targets, rng, indices)
        return self

    def predict(self, X):
        """Return continuous predictions with the same target rank as during fit."""
        predictions = self._predict_outputs(X)
        predictions = predictions * self.target_scale_ + self.target_mean_
        return predictions[:, 0] if self._single_target else predictions
